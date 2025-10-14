from src.model import analyze_sentiment

def test_positive_sentence():
    review = (
        "Very fast delivery. I ordered in the early afternoon and received it "
        "the at evening which was amazing considering I didn’t expect it "
        "until the next day. Amazing piece of kit, was super easy to install "
        "and set up. Works very well, I’m not good with installation so this "
        "was great for me, all pieces included including screws. Comes with "
        "instructions on how to set up the app and WiFi etc. the "
        "notifications make it easier for me to know if someone’s at my door "
        "and great to speak to courier to leave parcel when I’m not home. "
        "Feels secure and safe to have at my home and easy monitoring. Sound "
        "quality and motion detection is great, you can even set in the app "
        "how sensitive you want it too. Highly recommend."
    )
    result = analyze_sentiment(review)
    assert 'label' in result
    assert result['label'] == 'positive'

def test_negative_sentence():
    review = (
        "Dint receive the head wear and the cross,The size is not correct "
        "either"
    )
    result = analyze_sentiment(review)
    assert result['label'] == 'negative'
