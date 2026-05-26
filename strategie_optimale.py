from __future__ import annotations
import argparse
from dataclasses import dataclass
from functools import lru_cache
from random import Random
from statistics import mean
from typing import Callable, Sequence

import pulp


Gain = float
Strategie = Callable[["EtatJeu", "ConfigurationJeu", Random], int]


@dataclass(frozen=True)
class ConfigurationJeu:
    """Parametres modifiables du jeu Trolls et Chateaux."""

    nombre_cases: int = 7
    pierres_gauche: int = 15
    pierres_droite: int = 15
    position_initiale_troll: int | None = None

    def __post_init__(self) -> None:
        if self.nombre_cases < 3 or self.nombre_cases % 2 == 0:
            raise ValueError("Le nombre de cases doit etre impair et au moins egal a 3.")
        if self.pierres_gauche < 0 or self.pierres_droite < 0:
            raise ValueError("Les nombres de pierres doivent etre positifs ou nuls.")
        if self.position_initiale_troll is not None:
            if not 0 <= self.position_initiale_troll < self.nombre_cases:
                raise ValueError("La position initiale du troll est invalide.")

    @property
    def centre(self) -> int:
        return self.nombre_cases // 2

    @property
    def position_initiale(self) -> int:
        return self.centre if self.position_initiale_troll is None else self.position_initiale_troll


@dataclass(frozen=True)
class EtatJeu:
    """Etat complet d'une partie a un instant donne."""

    position_troll: int
    reserve_gauche: int
    reserve_droite: int


@dataclass(frozen=True)
class ResultatSimulation:
    parties: int
    victoires_gauche: int
    victoires_droite: int
    matchs_nuls: int
    score_moyen_gauche: float


def creer_etat_initial(configuration: ConfigurationJeu) -> EtatJeu:
    return EtatJeu(
        position_troll=configuration.position_initiale,
        reserve_gauche=configuration.pierres_gauche,
        reserve_droite=configuration.pierres_droite,
    )


def est_chateau(etat: EtatJeu, configuration: ConfigurationJeu) -> bool:
    return etat.position_troll == 0 or etat.position_troll == configuration.nombre_cases - 1


def valeur_position_finale(position_troll: int, configuration: ConfigurationJeu) -> Gain:
    """Gain du joueur gauche: 1 victoire, -1 defaite, 0 match nul."""

    if position_troll == configuration.centre:
        return 0.0
    if position_troll < configuration.centre:
        return -1.0
    return 1.0


def valeur_si_partie_arretee(etat: EtatJeu, configuration: ConfigurationJeu) -> Gain:
    """Applique la regle de fin quand au moins une reserve est vide."""

    if etat.reserve_gauche > 0 and etat.reserve_droite > 0:
        raise ValueError("Les deux joueurs ont encore des pierres.")

    if etat.reserve_gauche == 0 and etat.reserve_droite == 0:
        return valeur_position_finale(etat.position_troll, configuration)

    if etat.reserve_gauche == 0:
        position_finale = etat.position_troll - etat.reserve_droite
    else:
        position_finale = etat.position_troll + etat.reserve_gauche

    position_finale = max(0, min(configuration.nombre_cases - 1, position_finale))
    return valeur_position_finale(position_finale, configuration)


def appliquer_tour(
    etat: EtatJeu,
    pierres_gauche: int,
    pierres_droite: int,
    configuration: ConfigurationJeu,
) -> EtatJeu | Gain:
    """Joue un tour simultane et renvoie l'etat suivant ou le gain final."""

    if pierres_gauche < 1 or pierres_droite < 1:
        raise ValueError("Chaque joueur doit lancer au moins une pierre.")
    if pierres_gauche > etat.reserve_gauche or pierres_droite > etat.reserve_droite:
        raise ValueError("Un joueur lance plus de pierres que sa reserve.")

    nouvelle_position = etat.position_troll
    if pierres_gauche > pierres_droite:
        nouvelle_position += 1
    elif pierres_droite > pierres_gauche:
        nouvelle_position -= 1

    prochaine_etat = EtatJeu(
        position_troll=max(0, min(configuration.nombre_cases - 1, nouvelle_position)),
        reserve_gauche=etat.reserve_gauche - pierres_gauche,
        reserve_droite=etat.reserve_droite - pierres_droite,
    )

    if est_chateau(prochaine_etat, configuration):
        return valeur_position_finale(prochaine_etat.position_troll, configuration)
    if prochaine_etat.reserve_gauche == 0 or prochaine_etat.reserve_droite == 0:
        return valeur_si_partie_arretee(prochaine_etat, configuration)
    return prochaine_etat


def resoudre_jeu_matriciel(matrice: Sequence[Sequence[Gain]]) -> tuple[Gain, tuple[float, ...]]:
    """Resout le jeu a somme nulle et retourne valeur + strategie du joueur ligne."""

    if not matrice or not matrice[0]:
        raise ValueError("La matrice de gain doit etre non vide.")

    nombre_lignes = len(matrice)
    nombre_colonnes = len(matrice[0])
    probleme = pulp.LpProblem("jeu_troll", pulp.LpMaximize)

    probabilites = [pulp.LpVariable(f"p_{i}", lowBound=0.0) for i in range(nombre_lignes)]
    valeur = pulp.LpVariable("v")

    probleme += valeur
    probleme += pulp.lpSum(probabilites) == 1.0

    for colonne in range(nombre_colonnes):
        esperance_colonne = pulp.lpSum(
            probabilites[ligne] * float(matrice[ligne][colonne])
            for ligne in range(nombre_lignes)
        )
        probleme += esperance_colonne >= valeur

    statut = probleme.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[statut] != "Optimal":
        raise RuntimeError(f"Resolution impossible: {pulp.LpStatus[statut]}")

    distribution = tuple(max(0.0, float(pulp.value(p) or 0.0)) for p in probabilites)
    total = sum(distribution)
    if total == 0.0:
        distribution = tuple(1.0 / nombre_lignes for _ in range(nombre_lignes))
    else:
        distribution = tuple(p / total for p in distribution)

    return float(pulp.value(valeur)), distribution


@lru_cache(maxsize=None)
def valeur_et_strategie(
    position_troll: int,
    reserve_gauche: int,
    reserve_droite: int,
    nombre_cases: int,
) -> tuple[Gain, tuple[float, ...]]:
    """Calcule recursivement la strategie prudente du joueur gauche."""

    configuration = ConfigurationJeu(
        nombre_cases=nombre_cases,
        pierres_gauche=reserve_gauche,
        pierres_droite=reserve_droite,
        position_initiale_troll=position_troll,
    )
    etat = EtatJeu(position_troll, reserve_gauche, reserve_droite)

    if est_chateau(etat, configuration):
        return valeur_position_finale(position_troll, configuration), (1.0,)
    if reserve_gauche == 0 or reserve_droite == 0:
        return valeur_si_partie_arretee(etat, configuration), (1.0,)

    matrice: list[list[Gain]] = []
    for pierres_gauche in range(1, reserve_gauche + 1):
        ligne: list[Gain] = []
        for pierres_droite in range(1, reserve_droite + 1):
            resultat = appliquer_tour(etat, pierres_gauche, pierres_droite, configuration)
            if isinstance(resultat, EtatJeu):
                valeur_suivante, _ = valeur_et_strategie(
                    resultat.position_troll,
                    resultat.reserve_gauche,
                    resultat.reserve_droite,
                    nombre_cases,
                )
                ligne.append(valeur_suivante)
            else:
                ligne.append(resultat)
        matrice.append(ligne)

    return resoudre_jeu_matriciel(matrice)


def choisir_depuis_distribution(distribution: Sequence[float], rng: Random) -> int:
    """Tire un coup parmi 1..n avec les probabilites donnees."""

    seuil = rng.random()
    cumul = 0.0
    for indice, probabilite in enumerate(distribution, start=1):
        cumul += probabilite
        if seuil <= cumul:
            return indice
    return len(distribution)


def strategie_prudente_gauche(etat: EtatJeu, configuration: ConfigurationJeu, rng: Random) -> int:
    _, distribution = valeur_et_strategie(
        etat.position_troll,
        etat.reserve_gauche,
        etat.reserve_droite,
        configuration.nombre_cases,
    )
    return choisir_depuis_distribution(distribution, rng)


def strategie_prudente_droite(etat: EtatJeu, configuration: ConfigurationJeu, rng: Random) -> int:
    """Strategie prudente du joueur droite obtenue par symetrie."""

    etat_miroir = EtatJeu(
        position_troll=configuration.nombre_cases - 1 - etat.position_troll,
        reserve_gauche=etat.reserve_droite,
        reserve_droite=etat.reserve_gauche,
    )
    _, distribution = valeur_et_strategie(
        etat_miroir.position_troll,
        etat_miroir.reserve_gauche,
        etat_miroir.reserve_droite,
        configuration.nombre_cases,
    )
    return choisir_depuis_distribution(distribution, rng)


def strategie_aleatoire_gauche(etat: EtatJeu, _configuration: ConfigurationJeu, rng: Random) -> int:
    return rng.randint(1, etat.reserve_gauche)


def strategie_aleatoire_droite(etat: EtatJeu, _configuration: ConfigurationJeu, rng: Random) -> int:
    return rng.randint(1, etat.reserve_droite)


def strategie_econome_gauche(_etat: EtatJeu, _configuration: ConfigurationJeu, _rng: Random) -> int:
    return 1


def strategie_econome_droite(_etat: EtatJeu, _configuration: ConfigurationJeu, _rng: Random) -> int:
    return 1


def strategie_agressive_gauche(etat: EtatJeu, _configuration: ConfigurationJeu, _rng: Random) -> int:
    return max(1, etat.reserve_gauche // 2)


def strategie_agressive_droite(etat: EtatJeu, _configuration: ConfigurationJeu, _rng: Random) -> int:
    return max(1, etat.reserve_droite // 2)


def coup_valide(coup: object, reserve: int) -> bool:
    """Verifie qu'un coup correspond a un nombre entier de pierres jouable."""

    return type(coup) is int and 1 <= coup <= reserve


def simuler_partie(
    configuration: ConfigurationJeu,
    strategie_gauche: Strategie,
    strategie_droite: Strategie,
    rng: Random,
) -> Gain:
    """Simule une partie complete et renvoie le gain du joueur gauche."""

    etat = creer_etat_initial(configuration)

    while True:
        if est_chateau(etat, configuration):
            return valeur_position_finale(etat.position_troll, configuration)
        if etat.reserve_gauche == 0 or etat.reserve_droite == 0:
            return valeur_si_partie_arretee(etat, configuration)

        coup_gauche = strategie_gauche(etat, configuration, rng)
        coup_droite = strategie_droite(etat, configuration, rng)

        coup_gauche_invalide = not coup_valide(coup_gauche, etat.reserve_gauche)
        coup_droite_invalide = not coup_valide(coup_droite, etat.reserve_droite)

        if coup_gauche_invalide and coup_droite_invalide:
            return 0.0
        if coup_gauche_invalide:
            return -1.0
        if coup_droite_invalide:
            return 1.0

        resultat = appliquer_tour(etat, coup_gauche, coup_droite, configuration)
        if isinstance(resultat, EtatJeu):
            etat = resultat
        else:
            return resultat


def simuler_affrontement(
    configuration: ConfigurationJeu,
    strategie_gauche: Strategie,
    strategie_droite: Strategie,
    nombre_parties: int = 1000,
    graine: int = 0,
) -> ResultatSimulation:
    rng = Random(graine)
    gains = [
        simuler_partie(configuration, strategie_gauche, strategie_droite, rng)
        for _ in range(nombre_parties)
    ]
    return ResultatSimulation(
        parties=nombre_parties,
        victoires_gauche=sum(1 for gain in gains if gain > 0),
        victoires_droite=sum(1 for gain in gains if gain < 0),
        matchs_nuls=sum(1 for gain in gains if gain == 0),
        score_moyen_gauche=mean(gains),
    )


def afficher_distribution(distribution: Sequence[float]) -> None:
    for pierres, probabilite in enumerate(distribution, start=1):
        if probabilite > 1e-6:
            print(f"  {pierres:2d} pierre(s) : {probabilite:.4f}")


def afficher_resultat_simulation(nom: str, resultat: ResultatSimulation) -> None:
    print(f"{nom}:")
    print(f"  parties           : {resultat.parties}")
    print(f"  victoires gauche  : {resultat.victoires_gauche}")
    print(f"  victoires droite  : {resultat.victoires_droite}")
    print(f"  matchs nuls       : {resultat.matchs_nuls}")
    print(f"  score moyen gauche: {resultat.score_moyen_gauche:.3f}")


def construire_configuration_depuis_args(args: argparse.Namespace) -> ConfigurationJeu:
    return ConfigurationJeu(
        nombre_cases=args.cases,
        pierres_gauche=args.pierres_gauche,
        pierres_droite=args.pierres_droite,
        position_initiale_troll=args.position,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategie prudente du jeu Trolls et Chateaux")
    parser.add_argument("--cases", type=int, default=7)
    parser.add_argument("--pierres-gauche", type=int, default=15)
    parser.add_argument("--pierres-droite", type=int, default=15)
    parser.add_argument("--position", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()

    configuration = construire_configuration_depuis_args(args)
    etat_initial = creer_etat_initial(configuration)

    print("Configuration:", configuration)
    print("Etat initial:", etat_initial)

    valeur, distribution = valeur_et_strategie(
        etat_initial.position_troll,
        etat_initial.reserve_gauche,
        etat_initial.reserve_droite,
        configuration.nombre_cases,
    )
    print(f"Valeur prudente initiale pour le joueur gauche: {valeur:.6f}")
    print("Distribution prudente du premier coup:")
    afficher_distribution(distribution)

    print()
    afficher_resultat_simulation(
        "Prudente gauche contre aleatoire droite",
        simuler_affrontement(
            configuration,
            strategie_prudente_gauche,
            strategie_aleatoire_droite,
            args.simulations,
            args.graine,
        ),
    )
    afficher_resultat_simulation(
        "Prudente gauche contre agressive droite",
        simuler_affrontement(
            configuration,
            strategie_prudente_gauche,
            strategie_agressive_droite,
            args.simulations,
            args.graine,
        ),
    )
    afficher_resultat_simulation(
        "Prudente gauche contre prudente droite",
        simuler_affrontement(
            configuration,
            strategie_prudente_gauche,
            strategie_prudente_droite,
            args.simulations,
            args.graine,
        ),
    )
    print("Cache de calcul:", valeur_et_strategie.cache_info())


if __name__ == "__main__":
    main()
