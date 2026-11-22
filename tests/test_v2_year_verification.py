from hukou.year_verification import choose_year

def test_year_must_be_observed_not_created_from_digit_guesses():
    digits=[dict(index=3,value='4',confidence=.9,engine=e) for e in ('v5','en')]
    assert choose_year({'2020','2021'},digits)[0]==''
    assert choose_year({'2020','2024'},digits)[0]=='2024'

def test_one_recognizer_or_conflicting_digit_models_do_not_resolve_year():
    evidence=[dict(index=3,value='4',confidence=.99,engine='v5')]
    assert choose_year({'2020','2024'},evidence)[0]==''
    evidence += [dict(index=3,value=v,confidence=.9,engine=e) for v,e in (('4','en'),('0','v5'),('0','v6'))]
    assert choose_year({'2020','2024'},evidence)[0]==''
