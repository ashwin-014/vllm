import json
import time
import asyncio
import aiohttp


def sample_requests(text):
    prompt = """The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n"""
    prompt += text
    return prompt


async def send_request(
    prompt,
    api_url,
    best_of=1,
    use_beam_search=False,
    output_len=128,
    ignore_eos=False
):
    headers = {}
    pload = {
        "prompt": prompt,
        "n": 1,
        "best_of": best_of,
        "use_beam_search": use_beam_search,
        "temperature": 0,  # 0.0 if use_beam_search else 1.0,
        "top_p": 1.0,
        "max_tokens": output_len,
        "ignore_eos": ignore_eos,
        "stream": False,
        "output_control_config": {
            # "type": "select",
            # "schema": ["Manmohan Singh", "Narendra Modi", "Yogi", "Trump"]
            # "type": "json",
            # "schema": ["Manmohan Singh", "Narendra Modi", "Yogi", "Trump"]
            "type": "json",
            "schema": {
                "search_engine": "select: [\"Yahoo\", \"BuckBuckBo\"]",
                "fruit": "",
                "vegetable": "select: [\"Kovakkai\", \"Potato\", \"Tomato\"]",
            #     # "search_engine_3": "",
            #     # "search_engine_4": ""
            #     # "fruit3": "banana"
            }
        }
    }
    # "output_control_config": {
    #         "type": "selection",
    #         "schema": {"apple": "", "orange": "", "banana": ""}
    #     }

    timeout = aiohttp.ClientTimeout(total=3 * 100)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    print(f"output: {output}")
    return output


def main():
    api_url = "http://localhost:8000/generate"

    text = "List a one search engines and one fruit you know in the JSON format: "
    prompt = sample_requests(text)
    st = time.time()
    asyncio.run(send_request(prompt, api_url))
    print(time.time() - st)
    # await send_request(prompt, api_url)


if __name__ == "__main__":
    main()
