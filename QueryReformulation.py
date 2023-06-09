import numpy as np
import torch

# load dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# traing the trl for query expansion
from transformers import GPT2Tokenizer
from transformers import AutoTokenizer

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed
from trl.core import LengthSampler

from query_predictor import QueryReward
import pandas as pd
from tqdm import tqdm
from transformers.utils import logging

logging.set_verbosity_info()
import yaml

yaml_args = yaml.load(open("yaml_config/msmarco_config.yaml"), Loader=yaml.FullLoader)
qw = QueryReward(yaml_args["reward"], reward_type="post-retrieval",dataset="msmarco")


class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


config = PPOConfig(
    # "model_name": "/projects/futhark1/data/wzm289/code/RLSeq2SeqPytorch/scifact_results_best/checkpoint-1068",
    batch_size=64,
    steps=2000,
    # "seed": 123,
    log_with="wandb",
)


input_size = LengthSampler(yaml_args["txt_in_min_len"], yaml_args["txt_in_max_len"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# wandb.init(name=yaml_args["model_save_path"], project="gpt2-test", config=yaml_args)

# %%

# set seed before initializing value head for deterministic eval
set_seed(config.seed)
# %%
gpt2_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    yaml_args["model_name"]
).to(device)
gpt2_model_ref = AutoModelForCausalLMWithValueHead.from_pretrained(
    yaml_args["model_name"]
).to(device)

# wandb.watch(gpt2_model, log="all")

gpt2_tokenizer = GPT2Tokenizer.from_pretrained(
    "/projects/futhark1/data/wzm289/code/RLSeq2SeqPytorch/models/tokenizer"
)

# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_tokenizer.padding_side = "left"


class S2Sdataset(Dataset):
    def __init__(self, max_length=20, data_type="train"):
        self.input_ids = []
        self.attn_masks = []
        self.qids = []

        df_query = pd.read_csv(
            yaml_args["train_path"],
            sep="\t",
            names=["qid", "query"],
        )
        # print(df_query)
        self.df_train = df_query["query"]

        for e, item in enumerate(self.df_train):
            # tokenize
            encodings_dict = gpt2_tokenizer(
                item, truncation=True, max_length=max_length, padding="max_length"
            )
            self.qids.append(str(df_query["qid"].iloc[e]))
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]).to(device))
            self.attn_masks.append(
                torch.tensor(encodings_dict["attention_mask"]).to(device)
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "qids": self.qids[index],
            "query": self.df_train[index],
            "input_ids": self.input_ids[index],
            "attn_masks": self.attn_masks[index],
        }


train_dataset = S2Sdataset(data_type="train", max_length=20)

dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    drop_last=True,
)


ppo_trainer = PPOTrainer(
    config,
    gpt2_model,
    gpt2_model_ref,
    gpt2_tokenizer,
    dataset=train_dataset,
)

total_ppo_epochs = int(np.ceil(config.steps / config.batch_size))
print("total_ppo_epochs", total_ppo_epochs)
# %%
gen_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": gpt2_tokenizer.eos_token_id,
}

max_reward = 0
# for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(dataloader))):
# for epoch in range(total_ppo_epochs):
output_length_sampler = LengthSampler(
    yaml_args["txt_out_min_len"], yaml_args["txt_out_max_len"]
)

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):

    reward_all = []

    # for batch in dataloader:
    print("epoch:", epoch)
    # logs, timing = dict(), dict()
    # t0 = time.time()
    query_tensors = batch["input_ids"]
    #### Get response from gpt2
    # t = time.time()
    response_tensors = []
    query_tensor_list = []
    # for i in range(config.batch_size):
    for query in query_tensors:
        gen_len = output_length_sampler()
        response = gpt2_model.generate(
            query.unsqueeze(dim=0), max_new_tokens=gen_len, **gen_kwargs
        )
        query_tensor_list.append(query)
        response_tensors.append(response.squeeze()[-gen_len:])
        # print(gpt2_tokenizer.decode(response[0]))
    batch["response"] = [
        gpt2_tokenizer.decode(r.squeeze(), skip_special_tokens=True)
        for r in response_tensors
    ]
    # timing["time/get_response"] = time.time() - t

    #### Compute reward score
    # t = time.time()
    texts = [" ".join([q] * 5) + " " + r for q, r in zip(batch["query"], batch["response"])]
    print(texts)
    feedback_score = qw.get_reward_score(
        texts,
        None,
        qids=batch["qids"],
        source_text=None,
        data_type="train",
    )

    original_score = qw.get_reward_score(
        batch["query"],
        None,
        qids=batch["qids"],
        source_text=None,
        data_type="train",
    )

    print(feedback_score)
    reward_score = np.array(feedback_score) - np.array(original_score)

    rewards = torch.tensor(reward_score, device=device).detach()
    rewards_list = [r for r in rewards]

    # timing["time/get_sentiment_preds"] = time.time() - t

    #### Run PPO step
    # t = time.time()

    stats = ppo_trainer.step(query_tensor_list, response_tensors, rewards_list)
    ppo_trainer.log_stats(stats, batch, rewards)
    # timing["time/optimization"] = time.time() - t

    # #### Log everything
    # timing["time/epoch"] = time.time() - t0
    # table_rows = [
    #     list(r)
    #     for r in zip(
    #         batch["query"],
    #         batch["response"],
    #         original_score,
    #         rewards.cpu().tolist(),
    #     )
    # ]
    # logs.update(
    #     {
    #         "game_log": wandb.Table(
    #             columns=["query", "response", "original_score", "reward"],
    #             rows=table_rows,
    #         )
    #     }
    # )
    # logs.update(timing)
    # logs.update(stats)
    # logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy()
    reward_all.append(torch.mean(rewards).cpu().numpy())
    # logs["env/reward_std"] = torch.std(rewards).cpu().numpy()
    # logs["env/reward_dist"] = rewards.cpu().numpy()
    # wandb.log(logs)
    reward_now = np.mean(np.array(reward_all))
    print(reward_now)
    if reward_now >= max_reward:
        max_reward = reward_now
        print(reward_now)
        gpt2_model.save_pretrained(yaml_args["model_save_path"])
# wandb.finish()