import pandas as pd
import json
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FILE = '../data/items.csv'
OUTPUT_FILE = '../data/items_enriched_ai_turbo.csv'  # Sauvegarde sous data/ pour cohérence avec les notebooks
# Clé API lue via variable d'environnement (recommandé) ou fallback (non recommandé)
import os
API_KEY = os.getenv('OPENAI_API_KEY', 'confidential')  # ⚠️ Utiliser OPENAI_API_KEY dans l'env
BATCH_SIZE = 25  # Un peu plus gros
MAX_WORKERS = 10  # 10 requêtes simultanées (Attention aux Rate Limits !)

client = OpenAI(api_key=API_KEY)

# 1. Chargement
df = pd.read_csv(INPUT_FILE)

# Colonnes
for col in ['description', 'clean_author', 'category']:
    if col not in df.columns: df[col] = None

# On filtre ceux qui n'ont pas encore de description
indices_todo = df[df['description'].isna()].index.tolist()
print(f"Livres à traiter : {len(indices_todo)}")


# 2. Fonction unitaire (inchangée mais robuste)
def process_batch(batch_indices):
    prompt_items = []
    # On construit le texte pour le batch
    for idx in batch_indices:
        # On sécurise les valeurs pour éviter les erreurs de concaténation
        title = str(df.at[idx, 'Title'])
        auth = str(df.at[idx, 'Author']) if pd.notna(df.at[idx, 'Author']) else "Unknown"
        prompt_items.append(f"ID:{idx}|T:{title}|A:{auth}")

    prompt_text = "\n".join(prompt_items)

    # Prompt optimisé pour la vitesse (moins de tokens)
    system_msg = """Génère un JSON : liste d'objets {"id": 123, "desc": "Résumé français 1 phrase", "auth": "Auteur Nettoyé", "cat": "Genre"}. Devine si inconnu. JSON ONLY."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0  # Plus déterministe = plus rapide parfois
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        # Gestion des formats de réponse possibles
        if isinstance(data, dict):
            # Parfois l'IA met une clé "items" ou "books"
            keys = list(data.keys())
            return data[keys[0]] if keys else []
        return data

    except Exception as e:
        # Si erreur (souvent Rate Limit 429), on attend et on renvoie vide pour réessayer plus tard si besoin
        # Ici on return [] pour ne pas bloquer, mais l'idéal serait un retry
        return []


# 3. Exécution Parallèle
batches = [indices_todo[i:i + BATCH_SIZE] for i in range(0, len(indices_todo), BATCH_SIZE)]

print(f"--- Lancement Turbo : {len(batches)} lots avec {MAX_WORKERS} threads ---")

results_buffer = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # On lance tout
    future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

    # On récupère au fil de l'eau
    for future in tqdm(as_completed(future_to_batch), total=len(batches)):
        res = future.result()
        if res:
            results_buffer.extend(res)

# 4. Sauvegarde
print(f"Mise à jour du DataFrame avec {len(results_buffer)} résultats...")

# Optimisation de l'écriture (Map vs Iterrows)
updates = {
    'description': {},
    'clean_author': {},
    'category': {}
}

for item in results_buffer:
    try:
        idx = int(item.get('id'))
        updates['description'][idx] = item.get('desc')
        updates['clean_author'][idx] = item.get('auth')
        updates['category'][idx] = item.get('cat')
    except:
        pass

df['description'] = df['description'].fillna(pd.Series(updates['description']))
df['clean_author'] = df['clean_author'].fillna(pd.Series(updates['clean_author']))
df['category'] = df['category'].fillna(pd.Series(updates['category']))

df.to_csv(OUTPUT_FILE, index=False)
print("Terminé ! Vérifiez le fichier.")