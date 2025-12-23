import numpy as np

from phishdet.features import extract_lexical_features, build_feature_df


def test_extract_lexical_features_keys():
    url = "http://example.com/path?query=1"
    features = extract_lexical_features(url)
    expected_keys = {
        "url_len",
        "host_len",
        "path_len",
        "query_len",
        "dot_count",
        "dash_count",
        "underscore_count",
        "slash_count",
        "at_count",
        "qmark_count",
        "eq_count",
        "amp_count",
        "percent_count",
        "digit_count",
        "alpha_count",
        "has_https",
        "has_ip_host",
        "port_present",
        "subdomain_count",
        "token_count",
        "avg_token_len",
        "max_token_len",
        "shannon_entropy",
        "suspicious_keyword_hits",
    }
    assert expected_keys.issubset(set(features.keys()))


def test_build_feature_df_no_nan_inf():
    urls = ["http://example.com", "notaurl", "https://10.0.0.1/login"]
    df = build_feature_df(urls)
    assert df.shape[0] == 3
    assert not np.isnan(df.values).any()
    assert np.isfinite(df.values).all()


def test_extract_handles_malformed():
    broken_url = "::::://"
    features = extract_lexical_features(broken_url)
    assert isinstance(features, dict)
    assert all(np.isfinite(list(features.values())))
