import os
import random
from typing import Callable
import pandas as pd
from datasets import load_dataset
import pickle
from fuzzywuzzy import fuzz
import warnings
warnings.filterwarnings("ignore")

class PrecisionErrors:

    def __init__(self):

        self.number_letter_mapping = {
            "1": ["I", "l"],
            "2": ["Z"],
            "3": ["E"],
            "4": ["A"],
            "5": ["S"],
            "6": ["G"],
            "7": ["T"],
            "8": ["B"],
            "9": ["g", "q"],
            "I": ["1", "l"],
            "l": ["1", "I"],
            "Z": ["2"],
            "E": ["3"],
            "A": ["4"],
            "S": ["5"],
            "G": ["6"],
            "T": ["7"],
            "B": ["8"],
            "g": ["9", "q"],
            "q": ["9", "g"]
        }
        self.punctuations = [
            ".", ",", ";", ":", "!", "-", "'", '"'
        ]
        self.valid_func_mapping = {
            self.similar_character_confusion: self.valid_1,
            self.casing_mismatch: self.valid_2,
            self.missing_punctuation: self.valid_3,
            self.single_missing_character: self.valid_4,
            self.multiple_missing_characters: self.valid_5,
            self.transposition: self.valid_6,
            self.mishandled_whitespace: self.valid_7
        }

    def valid(self, func_name: Callable, text: str) -> bool:
        return self.valid_func_mapping[func_name](text)

    def similar_character_confusion(self, text: str) -> str:
        """
        This function will replace a random alphabet with a similar looking alphabet.
        :param text:
        :return: changed_text
        """

        number_letter_mapping = self.number_letter_mapping

        # randomly pick an alphabet from the text
        character = random.choice(text)
        while character not in number_letter_mapping:
            character = random.choice(text)

        # replace the character with a similar looking character
        similar_characters = number_letter_mapping[character]
        similar_character = random.choice(similar_characters)
        changed_text = text.replace(character, similar_character, 1)

        return changed_text

    def casing_mismatch(self, text: str) -> str:
        """
        This function will pick one alphabet from the text and change its case.
        :param text:
        :return: changed_text
        """

        # randomly pick an alphabet from the text
        alphabet = random.choice(text)
        while not alphabet.isalpha():
            alphabet = random.choice(text)

        # change the case of the alphabet
        changed_text = text.replace(alphabet, alphabet.swapcase(), 1)

        return changed_text

    def missing_punctuation(self, text: str) -> str:
        """
        This function will remove a random punctuation mark from the text.
        :param text:
        :return: changed_text
        """

        # list of punctuation marks
        punctuations_rep_space = [".", ",", ";", ":", "!", "-"]
        punctuations_rep_no_space = ["'", '"']
        total_punctuations = punctuations_rep_space + punctuations_rep_no_space

        # randomly pick a punctuation mark from the text
        punctuation = random.choice(total_punctuations)
        while punctuation not in text:
            punctuation = random.choice(total_punctuations)

        # replace the punctuation mark with a space or nothing based on the punctuation mark
        if punctuation in punctuations_rep_space:
            changed_text = text.replace(punctuation, " ", 1)
        else:
            changed_text = text.replace(punctuation, "", 1)

        return changed_text

    def single_missing_character(self, text: str) -> str:
        """
        This function will remove a random character from the middle of the text.
        :param text:
        :return: changed_text
        """

        # randomly pick a character from the text
        character = random.choice(text)

        # remove the character from the text
        changed_text = text.replace(character, "", 1)

        return changed_text

    def multiple_missing_characters(self, text: str) -> str:
        """
        This function will remove more than one random characters from the text.
        :param text:
        :return: changed_text
        """

        # remove between 2 and 5 characters from the text
        num_characters = random.randint(2, 5)
        while num_characters > len(text)-1:
            num_characters = random.randint(2, 5)

        changed_text = text
        for i in range(num_characters):
            changed_text = self.single_missing_character(changed_text)

        return changed_text

    def transposition(self, text: str) -> str:
        """
        This function will swap two side by side characters in the text randomly.
        :param text:
        :return: changed_text
        """

        # randomly pick a character from the text
        character = random.choice(text)

        # find the index of the character in the text
        index = text.index(character)
        while (index + 1 >= len(text) or (text[index]==" " or text[index+1]==" ") or text[index]==text[index+1]):
            character = random.choice(text)
            index = text.index(character)

        # swap the character with the next character
        changed_text = text[:index] + text[index + 1] + text[index] + text[index + 2:]

        return changed_text

    # def added_characters(self, text: str) -> str:
    #     return ""

    def mishandled_whitespace(self, text: str) -> str:
        """
        This function will randomly remove a whitespace from the text.
        :param text:
        :return: changed_text
        """

        # randomly pick index of any whitespace in the text
        index = random.randint(0, len(text) - 1)
        while text[index] != " ":
            index = random.randint(0, len(text) - 1)

        # remove the whitespace
        changed_text = text[:index] + text[index + 1:]

        return changed_text

    # Validation functions
    # Validation functions
    # Validation functions

    def valid_1(self, text: str) -> bool:

        allowed_chars = set(self.number_letter_mapping.keys())
        curr_chars = set(text)

        if len(allowed_chars.intersection(curr_chars)) == 0:
            return False
        return True

    def valid_2(self, text: str) -> bool:
        return text.isalpha()

    def valid_3(self, text: str) -> bool:

        for char in text:
            if char in self.punctuations:
                return True
        return False

    def valid_4(self, text: str) -> bool:
        return text.replace(" ", "") != ""

    def valid_5(self, text: str) -> bool:
        return (text.replace(" ", "")) != "" and (len(text) > 2)

    def valid_6(self, text: str) -> bool:
        return (text.replace(" ", "")) != "" and (len(text) > 1) and (set(text).__len__() > 3)

    def valid_7(self, text: str) -> bool:
        var = (text.replace(" ", "")) != "" and (" " in text)
        return var


def select_row(Perrors_obj, org_df, prec_errors, used_row_idx, error_type_usage, max_per_error_type):
    row_idx = random.randint(0, len(org_df) - 1)

    while row_idx in used_row_idx:
        row_idx = random.randint(0, len(org_df) - 1)
        #print("Row index already used. Selecting a new row index")

    # select a random error type
    error_type = random.randint(0, len(prec_errors) - 1)

    while error_type_usage[error_type] >= max_per_error_type:
        error_type = random.randint(0, len(prec_errors) - 1)

    # check if text is capable of getting this error induced
    valid = Perrors_obj.valid(
        func_name=prec_errors[error_type],
        text=org_df.iloc[row_idx]["answer"]
    )

    return row_idx, error_type, valid, error_type_usage


def induce_errors_randomly(org_df: pd.DataFrame, total_errors: int, max_per_error_type: int) -> pd.DataFrame:
    """
    This function will induce errors in the dataframe.
    :param org_df: Original dataframe
    :param total_errors: Number of errors to induce per error type

    :return: new_df
    """

    perrors_obj = PrecisionErrors()
    prec_errors = [
        perrors_obj.similar_character_confusion,
        perrors_obj.casing_mismatch,
        perrors_obj.missing_punctuation,
        perrors_obj.single_missing_character,
        perrors_obj.multiple_missing_characters,
        perrors_obj.transposition,
        # perrors_obj.added_characters,
        perrors_obj.mishandled_whitespace
    ]

    print("Inducing errors randomly")
    used_row_idx = set()
    error_type_usage = {i: 0 for i in range(len(prec_errors))}
    rows = []
    for i in range(total_errors):

        # select a random row
        row_idx, error_type, valid, error_type_usage = select_row(perrors_obj, org_df, prec_errors, used_row_idx,
                                                                  error_type_usage, max_per_error_type)
        #print(f"Row index: {row_idx}, Error type: {error_type}, Valid: {valid}")
        while not valid:
            row_idx, error_type, valid, error_type_usage = select_row(perrors_obj, org_df, prec_errors, used_row_idx,
                                                                      error_type_usage, max_per_error_type)

        error_type_usage[error_type] += 1

        # induce error
        old_text: str = str(org_df.iloc[row_idx]["answer"])
        new_text = prec_errors[error_type](old_text)

        # update the dataframe
        rows.append({
            "answer": new_text,
            "label": error_type,
            "cropped_image": org_df.iloc[row_idx]["cropped_image"],
            "old_text": old_text,
            "error_name": prec_errors[error_type].__name__,
            "distance": fuzz.ratio(old_text, new_text) / 100
        })

        used_row_idx.add(row_idx)

    print("Adding the remaining rows")
    for i in range(len(org_df)):
        if i not in used_row_idx:
            rows.append({
                "answer": org_df.iloc[i]["answer"],
                "label": -1,
                "cropped_image": org_df.iloc[i]["cropped_image"],
                "old_text": "",
                "error_name": "",
                "distance": 1
            })

    return pd.DataFrame(columns=["answer","label","cropped_image", "old_text","error_name","distance"], data=rows)


dataset = load_dataset("e-val/short_ocr_sentences")["train"]
error_ratio = 0.5
total_error_types = 7
total_errors = len(dataset) * error_ratio
error_fields_per_type = total_errors // total_error_types

df = dataset.to_pandas()
df = df.fillna("") \
    .replace("None", "") \
    .replace("nan", "") \
    .replace("null", "")

# drop all rows with answer value containing "03-" or "-03" or "-02" or "02-"
df = df[~df["answer"].str.contains("03-")]
df = df[~df["answer"].str.contains("-03")]
df = df[~df["answer"].str.contains("-02")]
df = df[~df["answer"].str.contains("02-")]
df = df[~df["answer"].str.contains("-01")]
df = df[~df["answer"].str.contains("01-")]
df = df[~df["answer"].str.contains("-00")]
df = df[~df["answer"].str.contains("00-")]

df = induce_errors_randomly(org_df=df, total_errors=int(total_errors), max_per_error_type=2500)

# save df as csv
# df.to_csv("data/induced_errors.csv", index=False)
with open('data/induced_errors_v1.pkl', 'wb') as file:
    pickle.dump(df, file)
    
# drop the image column
df = df.drop(columns=["cropped_image"])

# save df as csv
# df.to_csv("data/induced_errors_no_image.csv", index=False)
# with open('data/induced_errors_no_image.pkl', 'wb') as file:
#     pickle.dump(df, file)
    
print(df["error_name"].value_counts())