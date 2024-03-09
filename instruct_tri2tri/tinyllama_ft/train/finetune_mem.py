from instruct_tri2tri.tinyllama_ft.train.finetune import train

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
