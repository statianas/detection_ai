import os
import json
import re


def delete_files(directory):
    total_files = 0
    deleted_files = 0
    seen_files = {}

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            file_name, file_extension = os.path.splitext(file)

            delete = False

            # Удалить файлы с расширением .pdf
            if file_extension.lower() == '.pdf':
                delete = True

            # Удалить файлы меньше 8 КБ
            elif file_size < 8192:
                delete = True

            # Удалить файлы с одинаковым названием и размером
            elif (file_name, file_size) in seen_files:
                delete = True
            else:
                seen_files[(file_name, file_size)] = file_path

            # Удалить файлы с расширением .json, если d['link'] содержит 'researchgate'
            if not delete and file_extension.lower() == '.json':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'link' in data and 'researchgate' in data['link']:
                            delete = True
                except (json.JSONDecodeError, IOError):
                    pass
            if delete:
                try:
                    os.remove(file_path)
                    deleted_files += 1
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
            else:
                total_files += 1

    return total_files, deleted_files
