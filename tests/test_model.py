from src.model import analyze_sentiment

def test_positive_sentence():
    review = (
        "Amazon UK MAN1 is a very well-run and efficient warehouse. The site "
        "is easy to reach with clear directions, and the check-in process is "
        "smooth and well organised. Security is handled professionally, which "
        "keeps everything running safely and on time."
        "Staff are friendly, approachable, and always willing to assist, "
        "which makes the experience much better. Parking and access are "
        "straightforward, and the facility is kept clean and well managed."
        "Overall, a professional and efficient site – great experience every "
        "time."
    )
    result = analyze_sentiment(review)
    assert 'label' in result
    assert result['label'] == 'positive'

def test_negative_sentence():
    review = (
        "Bunch of crooks at this place item arrived back there on the 28th "
        "March have proof of delivery with a signature yet Amazon customer "
        "service say the item hadn’t arrived back yet.Now chasing up a refund "
        "for an £899 google pixel while some tea leaf is still working there "
        "stealing other items."
    )
    result = analyze_sentiment(review)
    assert result['label'] == 'negative'
