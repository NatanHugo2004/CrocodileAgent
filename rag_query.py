from pathlib import Path
import json, faiss, numpy as np
from build_index import embeddar_textos, carregar_documentos_txt, gerar_passagens, construir_indice
def carregar_indice(in_dir="storage"):
    index = faiss.read_index(f"{in_dir}/faiss.index")
    meta = json.loads(Path(f"{in_dir}/meta.json").read_text(encoding="utf-8"))
    return index, meta

def recuperar(query, k=5):
    index, meta = carregar_indice()
    qv = embeddar_textos([query])  # (1, d)
    D, I = index.search(qv, k)     # distâncias e índices
    resultados = []
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        p = meta[idx]
        resultados.append({
            "rank": rank,
            "score": float(score),
            "texto": p["texto"],
            "fonte": p["fonte"],
            "idx_local": p["idx_local"],
        })
    return resultados


def montar_contexto(resultados):
    ctx_linhas = []
    for r in resultados:
        cite = f"[{r['rank']}] ({r['fonte']} #{r['idx_local']})"
        ctx_linhas.append(f"{cite}\n{r['texto']}")
    return "\n\n".join(ctx_linhas)

def montar_prompt(pergunta, resultados):
    contexto = montar_contexto(resultados)
    prompt = f"""
    Você é um assistente que responde SOMENTE com base no CONTEXTO fornecido.
    Se a resposta não estiver no contexto, diga "Não encontrei a informação nos documentos.".

    Pergunta do usuário:
    {pergunta}

    CONTEXT0 (trechos recuperados):
    {contexto}

    Instruções:
    - Seja objetivo e inclua referências no formato [n] quando usar um trecho.
    - Se houver dados conflitantes, aponte a divergência.
    """
    return prompt.strip()

