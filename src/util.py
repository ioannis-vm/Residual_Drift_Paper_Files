"""
Utility functions for the project
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import sys
from datetime import datetime
from importlib.metadata import distributions

import git


def calculate_input_file_info(file_list: list[str]) -> str:
    """
    SHA256 checksum.

    Calculate a SHA256 checksum for each file in the file_list.
    Returns a descriptive string including file name,
    checksum, last modified date, and filesize.

    Returns
    -------
    str
      SHA256 checksum, last modified date, filesize.
    """
    file_info_strings = []

    for file_name in file_list:
        # Calculate individual file SHA256 checksum
        hash_sha256 = hashlib.sha256()
        with open(file_name, 'rb') as file:  # noqa: PTH123
            for byte_block in iter(lambda: file.read(4096), b''):
                hash_sha256.update(byte_block)

        file_checksum = hash_sha256.hexdigest()
        # Get last modified time and size of the file
        file_stats = os.stat(file_name)  # noqa: PTH116
        last_modified_date = datetime.fromtimestamp(file_stats.st_mtime).strftime(  # noqa: DTZ006
            '%Y-%m-%d %H:%M:%S'
        )
        file_size = float(file_stats.st_size)

        # Convert size to a human-friendly format
        suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        human_size = file_size
        i = 0
        while human_size >= 1024 and i < len(suffixes) - 1:  # noqa: PLR2004
            human_size /= 1024.0
            i += 1
        file_size_human = f'{human_size:.2f} {suffixes[i]}'

        # Combine information into a string for each file
        file_info = (
            f'    File: {os.path.basename(file_name)}\n'  # noqa: PTH119
            f'    Checksum: {file_checksum}\n'
            f'    Last Modified: {last_modified_date}\n'
            f'    Size: {file_size_human}\n'
        )
        file_info_strings.append(file_info)

    return '\n'.join(file_info_strings)


def store_info(
    path: (str | None) = None,
    input_data_paths: list[str] = [],  # noqa: B006
    seeds: list[int] = [],  # noqa: B006
) -> str:
    """
    Store metadata.

    Store metadata enabling reproducibility of results.

    Returns
    -------
    str
      Metadata contents.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # noqa: DTZ005
    metadata_content = f'Time: {timestamp}\n'

    # Check if the file already exists
    if path and os.path.isfile(path):  # noqa: PTH113
        timestamp_for_backup = datetime.now().strftime('%Y%m%d_%H%M%S')  # noqa: DTZ005
        backup_folder = os.path.join(  # noqa: PTH118
            os.path.dirname(path),  # noqa: PTH120
            'replaced_on_' + timestamp_for_backup,  # noqa: PTH120, RUF100
        )
        os.makedirs(backup_folder, exist_ok=True)  # noqa: PTH103
        shutil.move(path, backup_folder)
        metadata_content += f'Moved existing {path} in {backup_folder}\n'
        info_path = path + '.info'
        if os.path.isfile(info_path):  # noqa: PTH113
            shutil.move(info_path, backup_folder)
            metadata_content += f'Moved existing {info_path} in {backup_folder}\n'

    try:
        # Get the current repo SHA
        sha = git.Repo(os.getcwd()).head.commit.hexsha  # noqa: PTH109
        metadata_content += f'Repo SHA: {sha}\n'
    except git.exc.InvalidGitRepositoryError:
        metadata_content += 'Repo SHA: Git repo not found.\n'

    metadata_content += f'Hostname: {socket.gethostname()}\n'
    metadata_content += f'Python version: {sys.version}\n'

    # Get a list of installed packages and their versions using importlib.metadata
    installed_packages = [
        f'    {distribution.metadata["Name"]}=={distribution.version}'
        for distribution in distributions()
    ]
    installed_packages_str = '\n'.join(installed_packages)
    metadata_content += f'Installed packages:\n{installed_packages_str}\n'

    command_line_args = ' '.join(sys.argv)
    metadata_content += f'Command line arguments: {command_line_args}\n'

    if input_data_paths:
        file_info = calculate_input_file_info(input_data_paths)
        metadata_content += 'Input file information:\n'
        metadata_content += file_info

    if seeds:
        metadata_content += f'Random Seeds: {seeds}\n'

    if path:
        # Write contents to a file and return the original path.
        # Intended use: saving files directly on the file system
        os.makedirs(os.path.dirname(path), exist_ok=True)  # noqa: PTH103, PTH120
        with open(path + '.info', 'w', encoding='utf-8') as file:  # noqa: PTH123, FURB103
            file.write(metadata_content)
        return path
    # Otherwise retgurn the metadata as a string
    # Intended use: saving metadata in a database
    return metadata_content
