#!/usr/bin/env python3

import os
import sys
from typing import Dict, Any

def get_user_input(prompt: str, default: Any = None, use_defaults: bool = False) -> str:
    if use_defaults and default is not None:
        return str(default)
    default_display = f" [{default}]" if default is not None else ""
    response = input(f"{prompt}{default_display}: ").strip()
    if not response and default is not None:
        return str(default)
    return response

def create_config(use_defaults: bool = False) -> Dict[str, str]:
    config = {}
    
    config['ORCHESTRATOR_IMAGE'] = get_user_input(
        "Enter orchestrator image",
        "nineteenai/sn19:orchestrator-latest",
        use_defaults
    )
    
    config['IMAGE_SERVER_IMAGE'] = get_user_input(
        "Enter image server image",
        "nineteenai/sn19:image_server-latest",
        use_defaults
    )
    
    config['PORT'] = get_user_input(
        "Enter port number",
        "6920",
        use_defaults
    )
    
    refresh = get_user_input(
        "Auto-refresh local images? (True / False)",
        "True",
        use_defaults
    )
    if refresh in ('True', 'False'):
        config['REFRESH_LOCAL_IMAGES'] = '1' if refresh == 'True' else '0'

    if not use_defaults:
        device = get_user_input(
            "Enter GPU devices (e.g., 0 or 0,1,2) or press Enter to skip"
        )
        if device:
            config['DEVICE'] = device

    return config

def write_config(config: Dict[str, str], filename: str = ".vali.env") -> None:
    with open(filename, 'w') as f:
        f.write("# SN19 Configuration\n")
        f.write("# Generated by create_config.py\n\n")
        
        for key in ['ORCHESTRATOR_IMAGE', 'IMAGE_SERVER_IMAGE', 'PORT', 'REFRESH_LOCAL_IMAGES']:
            if key in config:
                value = config[key]
                if key in ['ORCHESTRATOR_IMAGE', 'IMAGE_SERVER_IMAGE']:
                    f.write(f'{key}="{value}"\n')
                else:
                    f.write(f'{key}={value}\n')
        
        f.write("\n# Optional Settings\n")
        for key in config:
            if key not in ['ORCHESTRATOR_IMAGE', 'IMAGE_SERVER_IMAGE', 'PORT', 'REFRESH_LOCAL_IMAGES']:
                value = config[key]
                f.write(f'{key}={value}\n')

def main():
    use_defaults = "--default" in sys.argv
    
    if not use_defaults:
        print("Press Enter to accept default values or input your own.\n")
    
    if os.path.exists(".vali.env") and not use_defaults:
        response = input(".vali.env already exists. Overwrite? (y/n): ").lower()
        if response != 'y':
            print("Configuration creation cancelled.")
            return
    
    config = create_config(use_defaults)
    write_config(config)
    
    if not use_defaults:
        print("\nConfiguration file .vali.env has been created successfully!")
        print("You can manually edit this file later if needed.")

if __name__ == "__main__":
    main()