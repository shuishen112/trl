from transformers import GPT2LMHeadModel
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer
import yaml

yaml_args = yaml.load(open("yaml_config/nq_config.yaml"), Loader=yaml.FullLoader)

model_save_path = yaml_args["model_save_path"]
tokenizer = GPT2Tokenizer.from_pretrained(
    "/projects/futhark1/data/wzm289/code/RLSeq2SeqPytorch/models/tokenizer"
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

fout = open(f"{model_save_path}/gpt_predict.txt", "w")


def test():
    model = GPT2LMHeadModel.from_pretrained(model_save_path).cuda()
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # df = pd.read_csv(
    #     "/projects/futhark1/data/wzm289/code/RLSeq2SeqPytorch/scifact/test.source",
    #     names=["query"],
    #     sep="\t",
    # )
    df = pd.read_csv(yaml_args["test_path"], names = ["query"], sep = "\t")
    print(df)
    for text in tqdm(df["query"].to_list()):
        prompt = text

        # note that we only want to make the parameters same as the bart, so we set the max_length of query to 20.

        generated = tokenizer(
            f"{prompt}", return_tensors="pt", max_length=20
        ).input_ids.cuda()

        # perform prediction
        sample_outputs = model.generate(
            generated,
            do_sample=False,
            top_k=50,
            max_length=40,
            top_p=0.90,
            temperature=0,
            num_return_sequences=0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # decode the predicted tokens into texts
        predicted_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
        predicted_text = predicted_text.strip().replace("\n", "")
        target = " ".join(predicted_text.split("<|pad|>")[-1].split(" ")[:12])
        fout.write(target + "\n")
        fout.flush()
        # print("predicted_text", predicted_text)


test()
