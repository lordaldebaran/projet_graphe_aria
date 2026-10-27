"""Vérification de l'environnement du projet (décodage quantique par GNN).

Usage : python check_env.py

Le script importe chaque bibliothèque nécessaire, affiche sa version, puis
exécute un mini test de bout en bout (circuit Stim, décodage PyMatching,
passe avant d'une couche de GNN). Il se termine par un message clair
indiquant si l'environnement est prêt.
"""

import importlib
import sys

# --- Paramètres du script (aucun nombre « magique » dans les fonctions) ---

# Bibliothèques à vérifier : nom d'import Python -> nom du paquet pip.
REQUIRED_LIBRARIES = {
    "stim": "stim",
    "pymatching": "pymatching",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "torch": "torch",
    "torch_geometric": "torch_geometric",
}

# Distance du code de surface utilisée pour le mini test (plus petit cas utile).
TEST_CODE_DISTANCE = 3
# Probabilité d'erreur physique utilisée pour le mini test.
TEST_NOISE_PROBABILITY = 0.005
# Nombre de tirages simulés pour le mini test (petit pour rester instantané).
TEST_NUMBER_OF_SHOTS = 200
# Nombre de caractéristiques par nœud pour le test du GNN : (x, y, t).
TEST_NODE_FEATURES = 3
# Dimension cachée de la couche GNN de test.
TEST_HIDDEN_DIMENSION = 8
# Nombre de nœuds du petit graphe de test du GNN.
TEST_NUMBER_OF_NODES = 5


def check_library(import_name):
    """Tente d'importer une bibliothèque et renvoie son statut.

    Entrée :
        import_name (str) : nom utilisé dans `import ...` (ex. "stim").
    Sortie :
        tuple (bool, str) : (importée avec succès ?, version ou message
        d'erreur).
    Traitement :
        importe le module avec importlib, lit son attribut `__version__`
        s'il existe ("inconnue" sinon), et capture toute exception d'import
        pour la renvoyer sous forme de texte au lieu d'interrompre le script.
    """
    try:
        module = importlib.import_module(import_name)
    except Exception as error:  # noqa: BLE001 (on veut tout signaler)
        return False, str(error)
    return True, getattr(module, "__version__", "inconnue")


def check_all_libraries():
    """Vérifie toutes les bibliothèques de REQUIRED_LIBRARIES et les affiche.

    Entrée :
        aucune (utilise la constante REQUIRED_LIBRARIES).
    Sortie :
        list[str] : noms des bibliothèques manquantes ou en erreur
        (liste vide si tout est correct).
    Traitement :
        appelle `check_library` pour chaque entrée, affiche une ligne
        « OK » ou « ECHEC » par bibliothèque, et collecte les échecs.
    """
    missing = []
    for import_name, pip_name in REQUIRED_LIBRARIES.items():
        success, info = check_library(import_name)
        status = "OK   " if success else "ECHEC"
        print(f"[{status}] {pip_name:<16} {info}")
        if not success:
            missing.append(pip_name)
    return missing


def run_stim_and_matching_test():
    """Simule un petit circuit bruité avec Stim et le décode avec PyMatching.

    Entrée :
        aucune (utilise les constantes TEST_*).
    Sortie :
        float : taux d'erreur logique observé sur le mini test.
    Traitement :
        génère un circuit de mémoire de code de surface, échantillonne les
        détecteurs et l'observable logique, construit le décodeur PyMatching
        à partir du modèle d'erreur du circuit, décode les tirages et renvoie
        la fraction de tirages où la prédiction diffère de l'observable.
    """
    import numpy as np
    import pymatching
    import stim

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=TEST_CODE_DISTANCE,
        rounds=TEST_CODE_DISTANCE,
        after_clifford_depolarization=TEST_NOISE_PROBABILITY,
        before_round_data_depolarization=TEST_NOISE_PROBABILITY,
        before_measure_flip_probability=TEST_NOISE_PROBABILITY,
        after_reset_flip_probability=TEST_NOISE_PROBABILITY,
    )
    sampler = circuit.compile_detector_sampler()
    syndrome, observables = sampler.sample(
        TEST_NUMBER_OF_SHOTS, separate_observables=True
    )
    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    prediction = matching.decode_batch(syndrome)
    return float(np.any(prediction != observables, axis=1).mean())


def run_gnn_forward_test():
    """Fait passer un petit graphe aléatoire dans une couche de GNN.

    Entrée :
        aucune (utilise les constantes TEST_*).
    Sortie :
        tuple (int, int) : forme (nombre de nœuds, dimension cachée) de la
        sortie de la couche, attendue égale à
        (TEST_NUMBER_OF_NODES, TEST_HIDDEN_DIMENSION).
    Traitement :
        crée des caractéristiques de nœuds aléatoires et un anneau d'arêtes
        entre nœuds consécutifs, applique une couche GraphConv de PyTorch
        Geometric, et renvoie la forme du résultat.
    """
    import torch
    from torch_geometric.nn import GraphConv

    features = torch.randn(TEST_NUMBER_OF_NODES, TEST_NODE_FEATURES)
    sources = torch.arange(TEST_NUMBER_OF_NODES)
    targets = (sources + 1) % TEST_NUMBER_OF_NODES
    edge_index = torch.stack([sources, targets])
    layer = GraphConv(TEST_NODE_FEATURES, TEST_HIDDEN_DIMENSION)
    return tuple(layer(features, edge_index).shape)


def main():
    """Exécute toutes les vérifications et affiche un verdict final.

    Entrée :
        aucune.
    Sortie :
        int : code de sortie du programme (0 si tout est prêt, 1 sinon).
    Traitement :
        affiche la version de Python, vérifie les imports, puis (si tous
        sont corrects) lance les deux mini tests et compare la forme de la
        sortie du GNN à celle attendue.
    """
    print(f"Python {sys.version.split()[0]}")
    missing = check_all_libraries()
    if missing:
        print("\nBibliothèques à installer ou à réparer :", ", ".join(missing))
        return 1
    logical_rate = run_stim_and_matching_test()
    print(f"\nStim + PyMatching : taux d'erreur logique du test = {logical_rate:.3f}")
    shape = run_gnn_forward_test()
    expected = (TEST_NUMBER_OF_NODES, TEST_HIDDEN_DIMENSION)
    print(f"GNN : forme de sortie {shape} (attendue {expected})")
    if shape != expected:
        return 1
    print("\nEnvironnement prêt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
