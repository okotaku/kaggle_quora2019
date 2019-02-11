import re
import json
import string
from unicodedata import category, normalize


def preprocess(text):
    """
    preprocess text main steps
    """
    text = remove_space(text)
    text = clean_special_punctuations(text)
    text = clean_number(text)
    text = pre_clean_rare_words(text)
    text = decontracted(text)
    text = clean_latex(text)
    text = clean_misspell(text)
    text = spacing_punctuation(text)
    text = spacing_some_connect_words(text)
    text = clean_bad_case_words(text)
    text = clean_repeat_words(text)
    text = remove_space(text)

    return text


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', '\x9d', '\xad',
              '\xa0']
    for space in spaces:
        text = text.replace(space, ' ')
    text = text.strip()
    text = re.sub('\s+', ' ', text)

    return text


def clean_special_punctuations(text, file_path="json/rare_words_mapping.json"):
    with open(file_path) as f:
        special_punc_mappings = json.load(f)
    for punc in special_punc_mappings:
        if punc in text:
            text = text.replace(punc, special_punc_mappings[punc])
    text = _remove_diacritics(text)

    return text


def _remove_diacritics(s):
    return ''.join(
        c for c in normalize('NFKD', s.replace('ø', 'o').replace('Ø', 'O').replace('⁻', '-').replace('₋', '-'))
        if category(c) != 'Mn')


def clean_number(text):
    text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
    text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
    text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)

    return text


def pre_clean_rare_words(text, file_path="json/rare_words_mapping.json"):
    with open(file_path) as f:
        rare_words_mapping = json.load(f)
    for rare_word in rare_words_mapping:
        if rare_word in text:
            text = text.replace(rare_word, rare_words_mapping[rare_word])

    return text


def decontracted(text):
    text = re.sub(r"(W|w)on(\'|\’)t ", "will not ", text)
    text = re.sub(r"(C|c)an(\'|\’)t ", "can not ", text)
    text = re.sub(r"(Y|y)(\'|\’)all ", "you all ", text)
    text = re.sub(r"(Y|y)a(\'|\’)ll ", "you all ", text)

    text = re.sub(r"(I|i)(\'|\’)m ", "i am ", text)
    text = re.sub(r"(A|a)in(\'|\’)t ", "is not ", text)
    text = re.sub(r"n(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)re ", " are ", text)
    text = re.sub(r"(\'|\’)s ", " is ", text)
    text = re.sub(r"(\'|\’)d ", " would ", text)
    text = re.sub(r"(\'|\’)ll ", " will ", text)
    text = re.sub(r"(\'|\’)t ", " not ", text)
    text = re.sub(r"(\'|\’)ve ", " have ", text)

    return text


def clean_latex(text):
    """
    convert r"[math]\vec{x} + \vec{y}" to English
    """
    text = re.sub(r'\[math\]', ' LaTex math ', text)
    text = re.sub(r'\[\/math\]', ' LaTex math ', text)
    text = re.sub(r'\\', ' LaTex ', text)

    pattern_to_sub = {
        r'\\mathrm': ' LaTex math mode ',
        r'\\mathbb': ' LaTex math mode ',
        r'\\boxed': ' LaTex equation ',
        r'\\begin': ' LaTex equation ',
        r'\\end': ' LaTex equation ',
        r'\\left': ' LaTex equation ',
        r'\\right': ' LaTex equation ',
        r'\\(over|under)brace': ' LaTex equation ',
        r'\\text': ' LaTex equation ',
        r'\\vec': ' vector ',
        r'\\var': ' variable ',
        r'\\theta': ' theta ',
        r'\\mu': ' average ',
        r'\\min': ' minimum ',
        r'\\max': ' maximum ',
        r'\\sum': ' + ',
        r'\\times': ' * ',
        r'\\cdot': ' * ',
        r'\\hat': ' ^ ',
        r'\\frac': ' / ',
        r'\\div': ' / ',
        r'\\sin': ' Sine ',
        r'\\cos': ' Cosine ',
        r'\\tan': ' Tangent ',
        r'\\infty': ' infinity ',
        r'\\int': ' integer ',
        r'\\in': ' in ',
    }
    pattern_dict = {k.strip('\\'): v for k, v in pattern_to_sub.items()}
    patterns = pattern_to_sub.keys()
    pattern_re = re.compile('(%s)' % '|'.join(patterns))

    def _replace(match):
        """
        reference: https://www.kaggle.com/hengzheng/attention-capsule-why-not-both-lb-0-694 # noqa
        """
        try:
            word = pattern_dict.get(match.group(0).strip('\\'))
        except KeyError:
            word = match.group(0)
            print('!!Error: Could Not Find Key: {}'.format(word))

        return word

    return pattern_re.sub(_replace, text)


def clean_misspell(text, file_name="json/misspell_mapping.json"):
    with open(file_name) as f:
        misspell_mapping = json.load(f)
    for bad_word in misspell_mapping:
        if bad_word in text:
            text = text.replace(bad_word, misspell_mapping[bad_word])

    return text


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    regular_punct = list(string.punctuation)
    extra_punct = [',', '.', '"', ':', ')', '(', '!', '?', '|', ';', "'", '$', '&',
                   '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£',
                   '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›',
                   '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”',
                   '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾',
                   '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼',
                   '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
                   'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
                   '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø',
                   '¹', '≤', '‡', '√', '«', '»', '´', 'º', '¾', '¡', '§', '£', '₤']
    all_punct = list(set(regular_punct + extra_punct))
    # do not spacing - and .
    all_punct.remove('-')
    all_punct.remove('.')
    for punc in all_punct:
        if punc in text:
            text = text.replace(punc, f' {punc} ')

    return text


def clean_bad_case_words(text, file_name="json/bad_case_words.json"):
    with open(file_name) as f:
        bad_case_words = json.load(f)
    for bad_word in bad_case_words:
        if bad_word in text:
            text = text.replace(bad_word, bad_case_words[bad_word])

    return text


def spacing_some_connect_words(text, file_name="json/misspell_mapping.json"):
    """
    'Whyare' -> 'Why are'
    """
    mis_connect_list = ['(W|w)hat', '(W|w)hy', '(H|h)ow', '(W|w)hich', '(W|w)here', '(W|w)ill']
    mis_connect_re = re.compile('(%s)' % '|'.join(mis_connect_list))
    with open(file_name) as f:
        mis_spell_mapping = json.load(f)
    ori = text
    for error in mis_spell_mapping:
        if error in text:
            text = text.replace(error, mis_spell_mapping[error])

    # what
    text = re.sub(r" (W|w)hat+(s)*[A|a]*(p)+ ", " WhatsApp ", text)
    text = re.sub(r" (W|w)hat\S ", " What ", text)
    text = re.sub(r" \S(W|w)hat ", " What ", text)
    # why
    text = re.sub(r" (W|w)hy\S ", " Why ", text)
    text = re.sub(r" \S(W|w)hy ", " Why ", text)
    # How
    text = re.sub(r" (H|h)ow\S ", " How ", text)
    text = re.sub(r" \S(H|h)ow ", " How ", text)
    # which
    text = re.sub(r" (W|w)hich\S ", " Which ", text)
    text = re.sub(r" \S(W|w)hich ", " Which ", text)
    # where
    text = re.sub(r" (W|w)here\S ", " Where ", text)
    text = re.sub(r" \S(W|w)here ", " Where ", text)
    #
    text = mis_connect_re.sub(r" \1 ", text)
    text = text.replace("What sApp", 'WhatsApp')

    text = remove_space(text)

    return text


def clean_repeat_words(text):
    text = text.replace("img", "ing")

    text = re.sub(r"(I|i)(I|i)+ng", "ing", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+y", "lly", text)
    text = re.sub(r"(A|a)(A|a)(A|a)+", "a", text)
    text = re.sub(r"(C|c)(C|c)(C|c)+", "cc", text)
    text = re.sub(r"(D|d)(D|d)(D|d)+", "dd", text)
    text = re.sub(r"(E|e)(E|e)(E|e)+", "ee", text)
    text = re.sub(r"(F|f)(F|f)(F|f)+", "ff", text)
    text = re.sub(r"(G|g)(G|g)(G|g)+", "gg", text)
    text = re.sub(r"(I|i)(I|i)(I|i)+", "i", text)
    text = re.sub(r"(K|k)(K|k)(K|k)+", "k", text)
    text = re.sub(r"(L|l)(L|l)(L|l)+", "ll", text)
    text = re.sub(r"(M|m)(M|m)(M|m)+", "mm", text)
    text = re.sub(r"(N|n)(N|n)(N|n)+", "nn", text)
    text = re.sub(r"(O|o)(O|o)(O|o)+", "oo", text)
    text = re.sub(r"(P|p)(P|p)(P|p)+", "pp", text)
    text = re.sub(r"(Q|q)(Q|q)+", "q", text)
    text = re.sub(r"(R|r)(R|r)(R|r)+", "rr", text)
    text = re.sub(r"(S|s)(S|s)(S|s)+", "ss", text)
    text = re.sub(r"(T|t)(T|t)(T|t)+", "tt", text)
    text = re.sub(r"(V|v)(V|v)+", "v", text)
    text = re.sub(r"(Y|y)(Y|y)(Y|y)+", "y", text)
    text = re.sub(r"plzz+", "please", text)
    text = re.sub(r"(Z|z)(Z|z)(Z|z)+", "zz", text)

    return text
