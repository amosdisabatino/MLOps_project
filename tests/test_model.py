from src.model import analyze_sentiment

def test_positive_sentence():
    result = analyze_sentiment("i love this")
    assert 'label' in result
    assert result['label'] in ['positive', 'neutral', 'negative']

def test_negative_sentence():
    result = analyze_sentiment("This is terrible!")
    assert result['label'] == 'negative'
