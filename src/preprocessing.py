import re
import nltk
import unidecode
import contractions
import wordninja
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
sw = stopwords.words("english")


def remove_emoji(text: str) -> str:
    """
    Auxiliary remove emojis.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without emojis.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def restore_hashtag(text: str) -> str:
    """
    Auxiliary function to the hashtags.
    E.g.: #sanantoniospursfan -> San Antonio Spurs fan

    Args:
        text (str): the text which will be restored.

    Returns:
        str: the text with the hashtags restored.
    """
    hashtags = re.findall(r"\#[^\s]+", text)

    if len(hashtags) > 0:
        for hashtag in hashtags:
            text = text.replace(hashtag, " ".join(wordninja.split(hashtag)))

    return text


def remove_newline(text: str) -> str:
    """
    Auxiliary function to remove newline characters.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without newline characters.
    """
    text = re.sub(r"\\n", " ", text)
    return text


def remove_usernames(text: str) -> str:
    """
    Auxiliary function to remove usernames within the text.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without usernames.
    """
    text = re.sub(r"\@[^\s]+", " ", text)
    return text


def format_contractions(text: str) -> str:
    """
    Auxiliary function to replace contractions with its formatted words.

    Args:
        text (str): the text which will be formatted.

    Returns:
        str: the text without contractions.
    """
    text = [contractions.fix(word) for word in text.split()]
    text = " ".join(text)
    return text


def remove_link(text: str) -> str:
    """
    Auxiliary function to remove website links.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text with all links removed.
    """
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return text


def keep_only_letters(text: str) -> str:
    """
    Auxiliary function to remove digits and special characters.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text with only letters.
    """
    text = re.sub(r"[^a-z\s]+", " ", text)
    return text


def remove_single_letters(text: str) -> str:
    """
    Auxiliary function to remove single isolated letters.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without single isolated letters.
    """
    text = re.sub(r"\b[a-z]\b", " ", text)
    return text


def remove_accent(text: str) -> str:
    """
    Auxiliary function to replace letters with accent to its version
    without the accent.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without accents.
    """
    return unidecode.unidecode(text)


def remove_stopwords(text: str) -> str:
    """
    Auxiliary function to remove stopwords from the text.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without stopwords.
    """
    text = [word for word in text.split() if word not in sw]
    text = " ".join(text)
    return text


def remove_extra_white_spaces(text: str) -> str:
    """
    Auxiliary function to remove extra white spaces.

    Args:
        text (str): the text which will be cleaned.

    Returns:
        str: the text without extra white spaces.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def lowercase(text: str) -> str:
    """
    Auxiliary function to lowercase the text.

    Args:
        text (str): the text which will be lowercased.

    Returns:
        str: the lowercase text.
    """
    return text.lower()


def preprocessing(text: str) -> str:
    """
    The main function responsible for preprocessing step.

    Args:
        text (str): the text which will be preprocessed/cleaned.

    Returns:
        str: the preprocessed/cleaned text.
    """
    text = lowercase(text)
    text = remove_link(text)
    text = remove_usernames(text)
    text = remove_emoji(text)
    text = restore_hashtag(text)
    text = format_contractions(text)
    text = remove_accent(text)
    text = keep_only_letters(text)
    text = remove_stopwords(text)
    text = remove_single_letters(text)
    text = remove_newline(text)
    text = remove_extra_white_spaces(text)
    return text
