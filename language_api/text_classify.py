from google.cloud import language_v1 as language


# Content Classification - Classify the string passed into categories and confidence score

def classify_text(text):
    class_list = []
    conf_list = []

    client = language.LanguageServiceClient()

    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

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
