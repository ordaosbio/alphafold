"""
Helper module for adding initial guess PDB support to AlphaFold.

This provides working implementations of different strategies to incorporate
an initial structural guess into AlphaFold predictions.
"""

import os
import numpy as np
from typing import Dict, Any, Optional
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import model
import logging
import jax.numpy as jnp

def parse_initial_guess(all_atom_positions) -> jnp.ndarray:
    '''
    Given a numpy array of all atom positions, return a jax array of the initial guess
    '''

    list_all_atom_positions = np.split(all_atom_positions, all_atom_positions.shape[0])

    templates_all_atom_positions = []

    # Initially fill with zeros
    for _ in list_all_atom_positions:
        templates_all_atom_positions.append(jnp.zeros((residue_constants.atom_type_num, 3)))

    for idx in range(len(list_all_atom_positions)):
        templates_all_atom_positions[idx] = list_all_atom_positions[idx][0] 

    return jnp.array(templates_all_atom_positions)


def initialize_prev_pos_with_guess(
    processed_feature_dict: Dict[str, Any],
    initial_guess_protein: protein.Protein,
    model_runner: model.RunModel,
) -> Dict[str, Any]:
    """
    Initialize AlphaFold's recycling 'prev_pos' feature with initial guess coordinates.
    
    This is the RECOMMENDED approach for incorporating an initial structural guess.
    It works within AlphaFold's existing recycling mechanism without modifying
    the model architecture.
    
    Requirements:
    - Model must have recycling enabled (num_recycle > 0 in config)
    - Initial guess sequence length must match target sequence
    
    Args:
        processed_feature_dict: Feature dictionary after model_runner.process_features()
        initial_guess_protein: Protein object loaded from initial guess PDB
        model_runner: The AlphaFold model runner instance
        
    Returns:
        Modified feature dictionary with prev_pos initialized
        
    Raises:
        ValueError: If sequence lengths don't match
    """
    # Validate sequence length
    target_length = processed_feature_dict['aatype'].shape[0]
    guess_length = len(initial_guess_protein.aatype)
    
    if target_length != guess_length:
        raise ValueError(
            f"Initial guess sequence length ({guess_length}) does not match "
            f"target sequence length ({target_length})"
        )
    
    # Get atom positions from initial guess
    atom_positions = initial_guess_protein.atom_positions  # Shape: [N_res, 37, 3]
    
    # AlphaFold's prev_pos uses pseudo-beta positions:
    # - CB (C-beta) for all residues except glycine
    # - CA (C-alpha) for glycine
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    
    # Determine which residues are glycine
    aatype = initial_guess_protein.aatype
    gly_idx = residue_constants.restype_order['G']
    is_gly = (aatype == gly_idx)
    
    # Select appropriate atoms (CB for non-GLY, CA for GLY)
    prev_pos = np.where(
        is_gly[:, None],
        atom_positions[:, ca_idx, :],
        atom_positions[:, cb_idx, :]
    )
    
    # Ensure correct dtype and shape
    prev_pos = prev_pos.astype(np.float32)
    
    if model_runner.multimer_mode:
        # For multimer, we need to handle the asym_id dimension
        # prev_pos shape should be [N_res, 3] but may need padding/masking
        logging.warning(
            "Multimer mode: Initial guess support may require additional handling "
            "for multi-chain features. Using basic implementation."
        )
        # You may need to split prev_pos by chain and add to each chain's features
        processed_feature_dict['prev_pos'] = jnp.array(prev_pos)
    else:
        # For monomer models, direct assignment works
        processed_feature_dict['prev_pos'] = jnp.array(prev_pos)
        
        # Optionally, also initialize other recycling features for stronger conditioning
        # These would need to be computed from the initial structure:
        # - prev_msa_first_row: [N_res, 256] - recycled single sequence representation
        # - prev_pair: [N_res, N_res, 128] - recycled pair representation
        # 
        # These are harder to compute without running the model once, so we skip them
        # and let the model compute them in the first recycling iteration
    
    logging.info(
        f"Initialized prev_pos with initial guess (shape: {prev_pos.shape})"
    )
    
    return processed_feature_dict


def validate_initial_guess(
    initial_guess_protein: protein.Protein,
    target_sequence: str,
    max_mismatch_fraction: float = 0.1
) -> bool:
    """
    Validate that initial guess is suitable for the target sequence.
    
    Args:
        initial_guess_protein: Protein object from initial guess
        target_sequence: Target amino acid sequence (1-letter code)
        max_mismatch_fraction: Maximum allowed fraction of mismatched residues
        
    Returns:
        True if validation passes, False otherwise
    """
    guess_seq = ''.join([
        residue_constants.restypes_with_x[aa] 
        for aa in initial_guess_protein.aatype
    ])
    
    if len(guess_seq) != len(target_sequence):
        logging.error(
            f"Length mismatch: initial guess has {len(guess_seq)} residues, "
            f"target has {len(target_sequence)} residues"
        )
        return False
    
    # Count mismatches
    mismatches = sum(1 for g, t in zip(guess_seq, target_sequence) if g != t)
    mismatch_fraction = mismatches / len(target_sequence)
    
    if mismatch_fraction > max_mismatch_fraction:
        logging.warning(
            f"High sequence mismatch: {mismatch_fraction:.1%} of residues differ "
            f"(threshold: {max_mismatch_fraction:.1%}). Initial guess may not be helpful."
        )
        return False
    
    logging.info(
        f"Initial guess validation passed: {mismatches}/{len(target_sequence)} "
        f"residues differ ({mismatch_fraction:.1%})"
    )
    
    return True


def load_and_validate_initial_guess(
    initial_guess_pdb_path: str,
    target_sequence: str,
) -> Optional[protein.Protein]:
    """
    Load initial guess PDB and validate it against target sequence.
    
    Args:
        initial_guess_pdb_path: Path to initial guess PDB file
        target_sequence: Target amino acid sequence (1-letter code)
        
    Returns:
        Protein object if successful, None otherwise
    """
    if not os.path.exists(initial_guess_pdb_path):
        logging.error(f"Initial guess PDB not found: {initial_guess_pdb_path}")
        return None
    
    try:
        with open(initial_guess_pdb_path, 'r') as f:
            pdb_string = f.read()
        
        initial_guess_protein = protein.from_pdb_string(pdb_string)
        logging.info(f"Loaded initial guess from {initial_guess_pdb_path}")
        
        # Validate
        if not validate_initial_guess(initial_guess_protein, target_sequence):
            return None
            
        return initial_guess_protein
        
    except Exception as e:
        logging.error(f"Failed to load initial guess: {e}")
        return None


def add_initial_guess_to_features(
    feature_dict: Dict[str, Any],
    processed_feature_dict: Dict[str, Any],
    initial_guess_pdb_path: str,
    model_runner: model.RunModel,
    strategy: str = 'prev_pos',
) -> Dict[str, Any]:
    """
    Unified interface for adding initial guess to AlphaFold features.
    
    Args:
        feature_dict: Raw feature dictionary from data pipeline
        processed_feature_dict: Feature dictionary after model_runner.process_features()
        initial_guess_pdb_path: Path to initial guess PDB file
        model_runner: The AlphaFold model runner instance
        strategy: Strategy to use ('prev_pos', 'template', or 'disabled')
        
    Returns:
        Modified processed_feature_dict with initial guess incorporated
    """
    if strategy == 'disabled' or initial_guess_pdb_path is None:
        return processed_feature_dict
    
    # Get target sequence from features
    target_aatype = feature_dict['aatype']
    target_sequence = ''.join([
        residue_constants.restypes_with_x[aa] 
        for aa in target_aatype
    ])
    
    # Load and validate initial guess
    initial_guess_protein = load_and_validate_initial_guess(
        initial_guess_pdb_path,
        target_sequence
    )
    
    if initial_guess_protein is None:
        logging.warning("Initial guess could not be loaded or validated. Proceeding without it.")
        return processed_feature_dict
    
    # Apply strategy
    if strategy == 'prev_pos':
        return initialize_prev_pos_with_guess(
            processed_feature_dict,
            initial_guess_protein,
            model_runner
        )
    elif strategy == 'template':
        # Template strategy would be implemented here
        # This requires more complex template featurization
        logging.error("Template strategy not yet implemented")
        return processed_feature_dict
    else:
        logging.error(f"Unknown initial guess strategy: {strategy}")
        return processed_feature_dict
