from google.cloud import language_v1 as language
import pandas as pd


# Content Classification - Classify the string passed into categories and confidence score

def classify_text(text_string):
    class_list = []
    conf_list = []

    client = language.LanguageServiceClient()

    document = language.Document(content=text_string, type_=language.Document.Type.PLAIN_TEXT)

    try:
        response = client.classify_text(document=document)
        categories = response.categories
        #         print(categories)

        for category in response.categories:
            class_list.append(category.name)
            conf_list.append(category.confidence)
    except:
        class_list.append('/Skipped')
        conf_list.append(0.0)

    return class_list, conf_list


def classify_result(text, sentence):

    # If sentence == 0, go for the whole text else break into sentences

    # Empty List to append results
    class_result = []
    confi_result = []

    # Split input string into lines
    if sentence == 1:
        lines = text.split('.')

        # For each line of input text, derive classification and confidence score
        for line in lines:

            class_list, conf_list = classify_text(line)
            #     print(class_list, conf_list)

            # A line can contain multiple classifications, list all of them with confidence score
            i = 0
            while len(class_list) > i:

                # If multiple subcategories, Extract the deepest category
                clas = class_list[i].split('/')
                class_result.append(clas[len(clas) - 1])
                confi_result.append(conf_list[i])
                i += 1
    else:
        class_list, conf_list = classify_text(text)
        clas = class_list[i].split('/')
        class_result.append(clas[len(clas) - 1])

    # Consolidate result into a dataframe
    result_df = pd.DataFrame({'Category': class_result, 'Confidence': confi_result})
    return result_df
