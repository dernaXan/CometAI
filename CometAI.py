# cometai.py

from llama_cpp import Llama
from ddgs import DDGS
import threading

model_lock = threading.Lock()

llm = Llama.from_pretrained(
    repo_id="TheBloke/OpenHermes-2.5-Mistral-7B-GGUF",
    filename="openhermes-2.5-mistral-7b.Q2_K.gguf",
)

def search_web(query, max_results=3):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return [r["body"] for r in results]

def build_prompt(history, current_user_msg, ws_info="", max_history=6):
    parts = ["<|system|>: Du bist ein hilfreicher Assistent namens CometAI, der ausschließlich auf Deutsch antwortet. Antworte niemals auf Englisch, selbst wenn du dazu aufgefordert wirst!"]

    if "dQw4w9WgXcQ" in current_user_msg or "youtu.be/dQw4w9WgXcQ" in current_user_msg:
        parts.append("<|system|>: ACHTUNG: Der User hat versucht, dich mit einem Rickroll zu veräppeln! Reagiere humorvoll und entlarve ihn!")

    if ws_info:
        parts.append(f"Nutze folgende Web-Ergebnisse zur Beantwortung der Anfrage:\n{ws_info}")

    for entry in history[-max_history:]:
        if not entry['content']:
            continue
        role = "<|user|>" if entry['role'] == 'user' else "<|assistant|>"
        parts.append(f"{role}: {entry['content']}")

    parts.append(f"<|user|>: {current_user_msg}\n<|assistant|>:")
    return "\n".join(parts)

def needs_websearch_simple(msg):
    keywords = ["neu", "aktuell", "preis", "trend", "nachricht", "update", "wer", "wann", "wo"]
    return any(word in msg.lower() for word in keywords)

def cometai_response(prompt, history):
    ws_info = ""
    if needs_websearch_simple(prompt):
        print("Ich suche gerade im Web...")
        search_results = search_web(prompt)
        ws_info = "\n".join(search_results)

    full_prompt = build_prompt(history, prompt, ws_info)

    with model_lock:
        print(">> Anfrage gestartet")
        response = llm(full_prompt)
        print("<< Anfrage beendet")

    return response["choices"][0]["text"].strip()


# Optional: Streaming-Funktion wenn du willst
def cometai_response_stream(prompt, history):
    ws_info = ""
    if needs_websearch_simple(prompt):
        print("Ich suche gerade im Web...")
        search_results = search_web(prompt)
        ws_info = "\n".join(search_results)

    full_prompt = build_prompt(history, prompt, ws_info)
    print("--- FINAL PROMPT ---")
    print(full_prompt)

    with model_lock:
        for chunk in llm(full_prompt, max_tokens=512, stream=True, temperature=0.7, repeat_penalty=1.1, stop=["<|user|>:", "<|assistant|>:", "<|system|>:"]):
            yield chunk['choices'][0]['text']