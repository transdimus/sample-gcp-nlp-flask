from datetime import datetime
import logging
import os
import pandas as pd
from flask import Flask, redirect, render_template, request

from google.cloud import datastore
from google.cloud import language_v1 as language


app = Flask(__name__)

@app.route("/")
def homepage():
    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # # Use the Cloud Datastore client to fetch information from Datastore
    # Query looks for all documents of the 'Sentences' kind, which is how we
    # store them in upload_text()
    query = datastore_client.query(kind="Sentences")
    text_entities = list(query.fetch())

    # # Return a Jinja2 HTML template and pass in text_entities as a parameter.
    return render_template("homepage.html", text_entities=text_entities)


@app.route("/upload", methods=["GET", "POST"])
def upload_text():
    text = request.form["text"]

    # Analyse sentiment using Sentiment API call
    sentiment = analyze_text_sentiment(text)[0].get('sentiment score')

    # classification = classify_text(text)[0]
    # class_df = pd.DataFrame(classification)
    # print(class_df)
    #
    # entities = analyze_entities(text)[0]
    # entities_df = entities.DataFrame(entities)
    # print(entities_df)

    # Assign a label based on the score
    overall_sentiment = 'unknown'
    if sentiment > 0:
        overall_sentiment = 'positive'
    if sentiment < 0:
        overall_sentiment = 'negative'
    if sentiment == 0:
        overall_sentiment = 'neutral'

    # Create a Cloud Datastore client.
    datastore_client = datastore.Client()

    # Fetch the current date / time.
    current_datetime = datetime.now()

    # The kind for the new entity. This is so all 'Sentences' can be queried.
    kind = "Sentences"

    # Create the Cloud Datastore key for the new entity.
    key = datastore_client.key(kind, 'sample_task')

    # Alternative to above, the following would store a history of all previous requests as no key
    # identifier is specified, only a 'kind'. Datastore automatically provisions numeric ids.
    # key = datastore_client.key(kind)

    # Construct the new entity using the key. Set dictionary values for entity
    entity = datastore.Entity(key)
    entity["text"] = text
    entity["timestamp"] = current_datetime
    entity["sentiment"] = overall_sentiment
    # entity["classification"] =
    # Save the new entity to Datastore.
    datastore_client.put(entity)

    # Redirect to the home page.
    return redirect("/")


@app.errorhandler(500)
def server_error(e):
    logging.exception("An error occurred during a request.")
    return (
        """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(
            e
        ),
        500,
    )


def analyze_text_sentiment(text):
    client = language.LanguageServiceClient()
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

    response = client.analyze_sentiment(document=document)

    sentiment = response.document_sentiment
    results = dict(
        text=text,
        score=f"{sentiment.score:.1%}",
        magnitude=f"{sentiment.magnitude:.1%}",
    )
    for k, v in results.items():
        print(f"{k:10}: {v}")

    # Get sentiment for all sentences in the document
    sentence_sentiment = []
    for sentence in response.sentences:
        item={}
        item["text"]=sentence.text.content
        item["sentiment score"]=sentence.sentiment.score
        item["sentiment magnitude"]=sentence.sentiment.magnitude
        sentence_sentiment.append(item)

    return sentence_sentiment

def sample_analyze_entity_sentiment(text_content):
    """
    Analyzing Entity Sentiment in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'Grapes are good. Bananas are bad.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entity_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Loop through entitites returned from the API
    for entity in response.entities:
        print(u"Representative name for the entity: {}".format(entity.name))
        # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        print(u"Entity type: {}".format(language_v1.Entity.Type(entity.type_).name))
        # Get the salience score associated with the entity in the [0, 1.0] range
        print(u"Salience score: {}".format(entity.salience))
        # Get the aggregate sentiment expressed for this entity in the provided document.
        sentiment = entity.sentiment
        print(u"Entity sentiment score: {}".format(sentiment.score))
        print(u"Entity sentiment magnitude: {}".format(sentiment.magnitude))
        # Loop over the metadata associated with entity. For many known entities,
        # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
        # Some entity types may have additional metadata, e.g. ADDRESS entities
        # may have metadata for the address street_name, postal_code, et al.
        for metadata_name, metadata_value in entity.metadata.items():
            print(u"{} = {}".format(metadata_name, metadata_value))

        # Loop over the mentions of this entity in the input document.
        # The API currently supports proper noun mentions.
        for mention in entity.mentions:
            print(u"Mention text: {}".format(mention.text.content))
            # Get the mention type, e.g. PROPER for proper noun
            print(
                u"Mention type: {}".format(language_v1.EntityMention.Type(mention.type_).name)
            )

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    print(u"Language of the text: {}".format(response.language))


def classify_text(text):
    client = language.LanguageServiceClient()
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)

    response = client.classify_text(document=document)
    return response


# Entity Analysis
def analyze_entities(text, debug=0):
    """
    Analyzing Entities in a String

    Args:
      text_content The text content to analyze
    """

    client = language.LanguageServiceClient()
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    response = client.analyze_entities(document=document)
    output = []

    # Loop through entitites returned from the API
    for entity in response.entities:
        item = {}
        item["name"] = entity.name
        item["type"] = language.Entity.Type(entity.type_).name
        item["Salience"] = entity.salience

        if debug:
            print(u"Representative name for the entity: {}".format(entity.name))

            # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
            print(u"Entity type: {}".format(language.Entity.Type(entity.type_).name))

            # Get the salience score associated with the entity in the [0, 1.0] range
            print(u"Salience score: {}".format(entity.salience))

        # Loop over the metadata associated with entity. For many known entities,
        # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
        # Some entity types may have additional metadata, e.g. ADDRESS entities
        # may have metadata for the address street_name, postal_code, et al.
        for metadata_name, metadata_value in entity.metadata.items():
            item[metadata_name] = metadata_value
            if debug:
                print(u"{}: {}".format(metadata_name, metadata_value))

        # Loop over the mentions of this entity in the input document.
        # The API currently supports proper noun mentions.
        if debug:
            for mention in entity.mentions:
                print(u"Mention text: {}".format(mention.text.content))
                # Get the mention type, e.g. PROPER for proper noun
                print(
                    u"Mention type: {}".format(language.EntityMention.Type(mention.type_).name)
                )
        output.append(item)

    # Get the language of the text, which will be the same as
    # the language specified in the request or, if not specified,
    # the automatically-detected language.
    if debug:
        print(u"Language of the text: {}".format(response.language))

    return (output)


if __name__ == "__main__":
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)
