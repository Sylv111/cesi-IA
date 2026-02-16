"""
Script pour fusionner les données d'enquête avec les données générales.

Ce script permet d'ajouter les colonnes d'enquête (EnvironmentSatisfaction, 
JobSatisfaction, WorkLifeBalance, JobInvolvement, PerformanceRating) depuis 
les fichiers manager_survey_data.csv et employee_survey_data.csv vers 
general_data.csv en utilisant EmployeeID comme clé de jointure.

Le fichier fusionné est sauvegardé dans le dossier "Output" pour préserver
les fichiers originaux dans le dossier "Source".

Le script est modulable et peut être utilisé avec d'autres fichiers CSV/Excel
qui contiennent une colonne EmployeeID.
"""

import pandas as pd
import os
from pathlib import Path


def merge_survey_data(
    general_data_path: str,
    employee_survey_path: str,
    manager_survey_path: str,
    output_path: str = None,
    employee_id_column: str = "EmployeeID",
    columns_to_add: list = None
):
    """
    Fusionne les données d'enquête avec les données générales.
    
    Parameters:
    -----------
    general_data_path : str
        Chemin vers le fichier CSV/Excel contenant les données générales
    employee_survey_path : str
        Chemin vers le fichier CSV/Excel contenant les données d'enquête employé
    manager_survey_path : str
        Chemin vers le fichier CSV/Excel contenant les données d'enquête manager
    output_path : str, optional
        Chemin de sortie pour le fichier fusionné. Si None, remplace le fichier général
    employee_id_column : str, default="EmployeeID"
        Nom de la colonne utilisée pour la jointure
    columns_to_add : list, optional
        Liste des colonnes à ajouter. Si None, utilise les colonnes par défaut
        
    Returns:
    --------
    pd.DataFrame
        DataFrame contenant les données fusionnées
    """
    
    # Colonnes par défaut à ajouter
    if columns_to_add is None:
        columns_to_add = [
            "EnvironmentSatisfaction",
            "JobSatisfaction", 
            "WorkLifeBalance",
            "JobInvolvement",
            "PerformanceRating"
        ]
    
    # Déterminer le chemin de sortie
    if output_path is None:
        # Par défaut, ne jamais écraser les fichiers Source
        output_path = str(Path(general_data_path).parent.parent / "Output" / "general_data_merged.csv")

    # Refuser toute sortie dans le dossier Source
    output_path_obj = Path(output_path).resolve()
    source_dir_obj = Path(general_data_path).resolve().parent
    if source_dir_obj in output_path_obj.parents or output_path_obj == Path(general_data_path).resolve():
        raise ValueError("Le fichier de sortie ne doit pas être dans le dossier Source.")

    # Créer le dossier parent si nécessaire
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Lecture du fichier général: {general_data_path}")
    # Lire le fichier général (support CSV et Excel)
    if general_data_path.endswith('.xlsx') or general_data_path.endswith('.xls'):
        general_df = pd.read_excel(general_data_path)
    else:
        general_df = pd.read_csv(general_data_path)
    
    print(f"Lecture du fichier d'enquête employé: {employee_survey_path}")
    # Lire le fichier d'enquête employé
    if employee_survey_path.endswith('.xlsx') or employee_survey_path.endswith('.xls'):
        employee_survey_df = pd.read_excel(employee_survey_path)
    else:
        employee_survey_df = pd.read_csv(employee_survey_path)
    
    print(f"Lecture du fichier d'enquête manager: {manager_survey_path}")
    # Lire le fichier d'enquête manager
    if manager_survey_path.endswith('.xlsx') or manager_survey_path.endswith('.xls'):
        manager_survey_df = pd.read_excel(manager_survey_path)
    else:
        manager_survey_df = pd.read_csv(manager_survey_path)
    
    # Vérifier que la colonne EmployeeID existe dans tous les fichiers
    for df_name, df in [("général", general_df), 
                         ("enquête employé", employee_survey_df),
                         ("enquête manager", manager_survey_df)]:
        if employee_id_column not in df.columns:
            raise ValueError(
                f"La colonne '{employee_id_column}' n'existe pas dans le fichier {df_name}"
            )
    
    # Supprimer les colonnes existantes si elles existent déjà (pour éviter les doublons)
    columns_to_remove = [col for col in columns_to_add if col in general_df.columns]
    if columns_to_remove:
        print(f"Suppression des colonnes existantes: {columns_to_remove}")
        general_df = general_df.drop(columns=columns_to_remove)
    
    # Supprimer aussi les colonnes avec suffixes _x et _y si elles existent
    suffix_columns_to_remove = []
    for col in columns_to_add:
        if f"{col}_x" in general_df.columns:
            suffix_columns_to_remove.append(f"{col}_x")
        if f"{col}_y" in general_df.columns:
            suffix_columns_to_remove.append(f"{col}_y")
    if suffix_columns_to_remove:
        print(f"Suppression des colonnes avec suffixes: {suffix_columns_to_remove}")
        general_df = general_df.drop(columns=suffix_columns_to_remove)
    
    # Sélectionner uniquement les colonnes à ajouter qui existent dans les fichiers d'enquête
    employee_columns = [col for col in columns_to_add if col in employee_survey_df.columns]
    manager_columns = [col for col in columns_to_add if col in manager_survey_df.columns]
    
    # Créer les DataFrames avec seulement EmployeeID et les colonnes à fusionner
    employee_to_merge = employee_survey_df[[employee_id_column] + employee_columns]
    manager_to_merge = manager_survey_df[[employee_id_column] + manager_columns]
    
    print(f"Fusion des colonnes d'enquête employé: {employee_columns}")
    # Fusionner avec les données d'enquête employé (sans suffixes car les colonnes n'existent pas)
    merged_df = general_df.merge(
        employee_to_merge,
        on=employee_id_column,
        how='left',
        suffixes=('', '_temp')
    )
    # Supprimer les colonnes temporaires avec suffixe si elles existent
    temp_cols = [col for col in merged_df.columns if col.endswith('_temp')]
    if temp_cols:
        merged_df = merged_df.drop(columns=temp_cols)
    
    print(f"Fusion des colonnes d'enquête manager: {manager_columns}")
    # Fusionner avec les données d'enquête manager (sans suffixes car les colonnes n'existent pas)
    merged_df = merged_df.merge(
        manager_to_merge,
        on=employee_id_column,
        how='left',
        suffixes=('', '_temp')
    )
    # Supprimer les colonnes temporaires avec suffixe si elles existent
    temp_cols = [col for col in merged_df.columns if col.endswith('_temp')]
    if temp_cols:
        merged_df = merged_df.drop(columns=temp_cols)
    
    # Vérifier les colonnes manquantes
    missing_columns = [col for col in columns_to_add if col not in merged_df.columns]
    if missing_columns:
        print(f"Attention: Les colonnes suivantes n'ont pas été trouvées: {missing_columns}")
    
    print(f"Sauvegarde du fichier fusionné: {output_path}")
    # Sauvegarder le résultat
    if output_path.endswith('.xlsx') or output_path.endswith('.xls'):
        merged_df.to_excel(output_path, index=False)
    else:
        merged_df.to_csv(output_path, index=False)
    
    print("Fusion terminee avec succes!")
    print(f"  - Nombre de lignes: {len(merged_df)}")
    print(f"  - Nombre de colonnes: {len(merged_df.columns)}")
    print(f"  - Colonnes ajoutees: {[col for col in columns_to_add if col in merged_df.columns]}")
    
    return merged_df


def main():
    """
    Fonction principale avec les chemins par défaut pour ce projet.
    """
    # Définir les chemins des fichiers
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    source_dir = project_dir / "Source"
    output_dir = project_dir / "Output"
    
    # Créer le dossier Output s'il n'existe pas
    output_dir.mkdir(exist_ok=True)
    
    general_data_path = source_dir / "general_data.csv"
    employee_survey_path = source_dir / "employee_survey_data.csv"
    manager_survey_path = source_dir / "manager_survey_data.csv"
    
    # Chemin de sortie dans le dossier Output
    output_path = output_dir / "general_data_merged.csv"
    
    # Vérifier que les fichiers existent
    for file_path in [general_data_path, employee_survey_path, manager_survey_path]:
        if not file_path.exists():
            raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
    
    # Exécuter la fusion
    merged_df = merge_survey_data(
        general_data_path=str(general_data_path),
        employee_survey_path=str(employee_survey_path),
        manager_survey_path=str(manager_survey_path),
        output_path=str(output_path)  # Sauvegarde dans le dossier Output
    )
    
    return merged_df


if __name__ == "__main__":
    try:
        merged_df = main()
        print("\nScript execute avec succes!")
    except Exception as e:
        print(f"\nErreur lors de l'execution: {e}")
        raise
