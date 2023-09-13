import fire
import time
from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        # Few shot prompt (providing a few examples before asking model to match);
        """Do the following records refer to the same real-world entity? Answer with "Yes" if affirmative and "No" otherwise.
        
        Record 1: name: Computas Subsidiary Corp, city: Lysaker, region: Akershus, country_code: NOR, short_description: Computas Subsidiary Corp is a Norwegian IT consulting company that provides services and solutions for business processes and collaboration 
        Record 2: name: Computas SC, country_code: Norway """,
    ]
    t = time.time()

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    elapsed = time.time() - t
    print(f"Elapsed time for 1 round of inference: {elapsed:.2f}s")

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
