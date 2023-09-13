from typing import Optional

import fire
import time

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [
        [{"role": "system", "content": "Your task is to decide wether Records 1 and 2 refer to the same real-world entity. Answer with 'Yes' if affirmative and 'No' otherwise."},
         {"role": "user", "content": """Do the following records refer to the same real-world entity?  
          Record 1: name: Computas Subsidiary Corp, city: Lysaker, region: Akershus, country_code: NOR, short_description: Computas Subsidiary Corp is a Norwegian IT consulting company that provides services and solutions for business processes and collaboration.
          Record 2: name: Computas SC, country_code: Norway"""},
          
        ],
        [{"role": "system", "content": "Your task is to decide wether Records 1 and 2 refer to the same real-world entity. Answer with 'Yes' if affirmative and 'No' otherwise."},
        {"role": "user", "content": """Do the following records refer to the same real-world entity?  
          Record 1: name: Rydoo, city: Mechelen, region: Antwerpen, country_code: Belgium, short_description: Rydoo is a leading business expense management solution that automates and streamlines processes for high-growth companies and enterprise.
          Record 2: name: Ryde Culture LTD, region: Minnesota, country_code: US"""},],
        [{"role": "system", "content": "Your task is to decide wether Records 1 and 2 refer to the same real-world entity. Answer with 'Yes' if affirmative and 'No' otherwise."},
        {"role": "user", "content": """Do the following records refer to the same real-world entity?
        Record 1: name: CrabTree CPA, region: Massachusetts, country_code: UNITED STATES, short_description: CrabTree CPA is a full-service accounting firm that provides tax, accounting, and consulting services to individuals and businesses.
        Record 2: name: Crabtree & ASSOCIATES city: Hyannis Port, region: Massachusetts, country_code: US, short_description: A full service accounting firm is called Crabtree & ASSOCIATES."""},
        ],

        [{"role": "system", "content": "Your task is to decide wether Records 1 and 2 refer to the same real-world entity. Answer with 'Yes' if affirmative and 'No' otherwise."},
            {"role": "user", "content": """Do the following records refer to the same real-world entity?
                Record 1: name: Carebuddy Co., region: Delhi, country_code: IN, short_description: UberHealth enables children living far away to manage their elderâ€™s well being from a distance.
                Record 2: name: CareBooker B-Corp city: Norwalk, region: CT, country_code: US"""},
        ],
            ]
    
    t = time.time()

    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    elapsed = (time.time() - t)/len(dialogs)
    print(f"Elapsed time for 1 round of inference: {elapsed:.2f}s")


    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)

# run with: torchrun --nproc_per_node 1 entity_matching_llama_2_7b_chat.py     --ckpt_dir llama-2-7b-chat/     --tokenizer_path tokenizer.model     --max_seq_len 512  --max_batch_size 4