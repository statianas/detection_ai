import subprocess
import os
import json
import paper_parse
import re
import orjson
import chardet
from progress.bar import IncrementalBar
from langdetect import detect_langs, LangDetectException
from tqdm import tqdm
from transliterate import translit, get_available_language_codes

directory = "Papers"

# Пример текста
text = """
<PAR>This is the first paragraph. It should be included in the
 first text segment. <PAR>This is the second paragraph.
  It should be included in the first text segment if it fits within 
  the 500 character limit. <PAR>This is the third paragraph. It will
   be included in the second text segment if it fits within the 500 character limit.
    This is additional text to ensure the length is sufficient for the demonstration. 
    <PAR>This is the fourth paragraph.</PAR>
"""
def starts_with_capital(text):
    # print(text[0])
    return text[0].isupper()

def remove_first_sentence(text):
    # Ищем первое предложение, заканчивающееся точкой
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    text = " ".join(sentences[1:])
    return text

def ensure_starts_with_capital(text):
    if starts_with_capital(text) or text[0].isnumeric():
        return text
    else:
        new_text = remove_first_sentence(text)
        return remove_first_sentence(new_text)

def is_asian_text(percentage, text):
    # Регулярное выражение для поиска китайских, корейских и японских иероглифов
    asian_characters = re.compile(
        r'[\u4e00-\u9fff]|[\u3000-\u303f]|[\u3040-\u309f]|[\u30a0-\u30ff]|[\uac00-\ud7af]'
    )
    total_chars = len(text)
    asian_chars = len(asian_characters.findall(text))

    return (asian_chars / total_chars) > percentage

def is_english_text(text, threshold=0.6):
    try:
        # detect_langs возвращает список языков с их вероятностями
        langs = detect_langs(text)
        for lang in langs:
            if lang.lang == 'en' and lang.prob >= threshold:
                return True
        return False
    except LangDetectException:
        return False


def remove_sentences_with_email(text):
    # Регулярное выражение для поиска e-mail
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # Фильтруем предложения, удаляя те, которые содержат e-mail
    filtered_sentences = [sentence for sentence in sentences if not email_pattern.search(sentence)]

    # Объединяем оставшиеся предложения обратно в текст
    return ' '.join(filtered_sentences)


def remove_last_sentence_without_period(text):
    # Убираем начальные и конечные пробелы
    text = text.strip()

    # Проверяем, заканчивается ли текст точкой
    if text.endswith('.'):
        return text

    # Разбиваем текст на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Убираем последнее предложение, если оно не заканчивается точкой
    if sentences and not sentences[-1].endswith('.'):
        sentences.pop()

    # Объединяем предложения обратно в текст
    return ' '.join(sentences)

translit_dict = {
    'А': 'A', 'а': 'a',
    'В': 'B', 'в': 'b',
    'Е': 'E', 'е': 'e',
    'К': 'K', 'к': 'k',
    'М': 'M', 'м': 'm',
    'Н': 'H', 'н': 'h',
    'О': 'O', 'о': 'o',
    'Р': 'P', 'р': 'p',
    'С': 'C', 'с': 'c',
    'Т': 'T', 'т': 't',
    'У': 'Y', 'у': 'y',
    'Х': 'X', 'х': 'x',
    'Г': 'r', 'г': 'r',
    'П': 'n', 'п': 'n',
}

def transliterate(text):
    return ''.join([translit_dict.get(char, char) for char in text])

def normalizing_text(text):
    words = text.split()
    if "abstract." == words[0].lower():
            text = " ".join(words[1:])
    if "abstract" == words[0].lower():
            text = " ".join(words[1:])
    if "abstract—" == words[0].lower():
            text = " ".join(words[1:])
    if "abstract:" == words[0].lower():
        text = " ".join(words[1:])
    if "introduction" == words[0].lower():
        text = " ".join(words[1:])
    if "—" == words[0][0].lower():
        text = " ".join(words[:][1:])
    if "*" == words[0][0].lower():
        text = " ".join(words[:][1:])
    # print(text)
    # try:
    #     text = ensure_starts_with_capital(text)
    # except:
    #     return ""
    # text = remove_sentences_with_email(text)
    text = remove_last_sentence_without_period(text)
    text = transliterate(text)
    return text

def has_letters_with_space(text):
    # Регулярное выражение для поиска двух подряд идущих букв через пробел
    pattern = re.compile(r'\b[A-Za-z]\s[A-Za-z]\b')
    return bool(pattern.search(text))


def is_normal_text(text):
    if '. . . . . . . . . . . .' in text:
        return False
    if is_asian_text(0.5, text):
        return False
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # if len(sentences[0]) < 6:
    #     return False
    if has_letters_with_space(text):
        return False
    if not is_english_text(text):
        return False

    return True



def split_text_into_segments(text, num_segments, is_normal, is_abstract):
    segments = []
    # word_limit = 200
    word_limit_min = 120
    word_limit_max = 200

    # Проверяем наличие тегов <PAR>
    if '<PAR>' in text:
        # paragraphs = re.split(r'<PAR>', text)
        paragraphs = text.split('<PAR>')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        # print(paragraphs)

        segment = ""
        current_length = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            segment = segment.strip()
            try:
                segment = ensure_starts_with_capital(segment)
                current_length = len(segment.split())
            except:
                segment = ""
                current_length = 0
            try:
                paragraph = ensure_starts_with_capital(paragraph)
            except:
                pass
            paragraph = remove_sentences_with_email(paragraph)
            paragraph_length = len(paragraph.split())
            if is_abstract and paragraph == paragraphs[0] and paragraph_length > 70:
                try:
                    paragraph = ensure_starts_with_capital(paragraph)
                    paragraph_length = len(paragraph.split())
                except:
                    segment = ""
                    current_length = 0
                if paragraph_length > word_limit_max:
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
                    while paragraph_length > word_limit_max:
                        sentences = sentences[:-1]
                        paragraph = ' '.join(sentences)
                        paragraph_length = len(paragraph.split())
                segments.append(paragraph.strip())
                continue
            # if word_limit_max > current_length + paragraph_length > word_limit_min:
            #     segment += " " + paragraph
            #     segments.append(segment.strip())
            #     segment = ""
            #     current_length = 0
                # else:
                #     segment = paragraph
                #     segments.append(segment.strip())
                #     segment = ""
                #     current_length = 0
            if current_length + paragraph_length > word_limit_max:
                if not segment:
                    try:
                        paragraph = ensure_starts_with_capital(paragraph)
                    except:
                        continue
                # paragraph = remove_sentences_with_email(paragraph)
                len_full = current_length + len(paragraph.split())
                sentences = re.split(r'(?<=[.!?])\s+', paragraph.strip())
                while len_full > word_limit_max:
                    sentences = sentences[:-1]
                    paragraph = ' '.join(sentences)
                    paragraph_length = len(paragraph.split())
                    len_full = current_length + paragraph_length
                segment += " " + paragraph
                segments.append(segment.strip())
                segment = " "
                current_length = 0
            elif word_limit_max > current_length + paragraph_length > word_limit_min:
                segment += " " + paragraph
                segments.append(segment.strip())
                segment = " "
                current_length = 0
            else:
                segment += " " + paragraph
                current_length = len(segment.split())
            if len(segments) == num_segments:
                break

        if segment and len(segments) < num_segments:
            segments.append(segment.strip())

    else:
        sentences = re.split(r'(?<=\.)\s+', text)
        segment = ""
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            segment = segment.strip()
            sentence_length = len(sentence.split())

            if word_limit_max > current_length + sentence_length > word_limit_min:
                if segment:
                    segment += " " + sentence
                    segments.append(segment.strip())
                    segment = ""
                    current_length = 0
                # else:
                #     segment = sentence
                #     segments.append(segment.strip())
                #     segment = ""
                #     current_length = 0

                # segment = sentence
                # current_length = sentence_length
            else:
                segment += " " + sentence
                try:
                    segment = ensure_starts_with_capital(segment)
                except:
                    segment = ""
                segment = remove_sentences_with_email(segment)
                current_length = len(segment.split())

            if len(segments) == num_segments:
                break

        if segment and len(segment.split()) < word_limit_max and len(segment.split()) > word_limit_min and len(segments) < num_segments:
            segments.append(segment.strip())

    if len(segments) < (3 if is_abstract else 2):
        is_normal = False

    return segments, is_normal

def create_article(title, link, year, texts):
    return {
        "title": title,
        "link": link,
        "year": year,
        "texts": texts
    }


articles = []
encodings = []

# bar = IncrementalBar('Countdown', max = 10300)

for root, dirs, files in os.walk(directory):
    for file in tqdm(files):
        # bar.next()
        file_path = os.path.join(root, file)
        file_name, file_extension = os.path.splitext(file)
        if file_name == ".DS_Store":
            continue
        # with open(file_path, 'rb') as f:
        #     raw_data = f.read()
        #     result = chardet.detect(raw_data)
        #     encoding = result['encoding']
        #     encodings.append(encoding)
        # print(set(encodings))
        # print(file_path)
        # Print detected encoding for verification
        # print(f"Detected encoding: {encoding}")
        # try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # except:
        #     print('error on ', file_path)
        #     with open(file_name, "r", encoding="Windows-1251") as f:
        #         data = json.load(f)
        #     continue
        title = data['title']
        link = data['link']
        year = data['year']

        if file_extension.lower() == '.json':
            is_abstract, text = paper_parse.parse(file_path)
            num_segments = 3 if is_abstract else 2
            is_normal = True
            segments, is_normal = split_text_into_segments(text, num_segments, is_normal, is_abstract)
            good_text = []
            # print(file_path)
            # print(f"Количество сегментов: {len(segments)}")
            if is_normal:
                for segment in segments:
                    if segment and is_normal_text(segment):
                        segment = normalizing_text(segment)
                        if segment and len(segment.split()) > 100:
                            good_text.append(segment)
            if good_text:
                articles.append(create_article(title, link, year, good_text))

json_data = json.dumps(articles, indent=4, ensure_ascii=False, separators=(',', ': '))

# Сохранение в файл
with open('articles.json', 'w', encoding='utf-8') as file:
    file.write(json_data)

