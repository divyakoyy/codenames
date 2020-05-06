import requests
import urllib
import json

"""
This script generates babelnet edge files for the specified word.
It goes to depth 3 hypernyms, and filters out automatic relations.
Note that this requires BABELNET_KEY to query the API 
"""

BABELNET_KEY = None


def get_synsets_from_lemma(word, limit):
    url = "https://babelnet.org/sparql/"
    queryString = """
    SELECT DISTINCT ?synset WHERE {{
        ?entries a lemon:LexicalEntry .
        ?entries lemon:sense ?sense .
        ?sense lemon:reference ?synset .
        ?entries rdfs:label "{word}"@en
    }} LIMIT {limit}
    """.format(limit=limit, word=word)
    query = queryString.replace(" ", "+")
    fmt = urllib.parse.quote("application/sparql-results+json".encode('UTF-8'), safe="")
    params = {
        "query": query,
        "format": fmt,
        "key": BABELNET_KEY,
    }
    payload_str = "&".join("%s=%s" % (k,v) for k,v in params.items())
    
    res = requests.get('?'.join([url, payload_str]))
    synsets = [
        'bn:' + r['synset']['value'].split('/')[-1].lstrip('s')
        for r in res.json()['results']['bindings']
    ]
    return synsets


def get_nonautomatic_hypernyms(results):
    return [
        result for result in results 
        if result['pointer']['isAutomatic'] is False 
        and result['pointer']['relationGroup'] == "HYPERNYM"
    ]


def get_outgoing_edges_json(synset_id):
    url = 'https://babelnet.io/v5/getOutgoingEdges'
    params = {
        'id': synset_id,
        'key': 'e3b6a00a-c035-4430-8d71-661cdf3d5837',
    }
    headers = {'Accept-Encoding': 'gzip'}
    res = requests.get(url=url, params=params, headers=headers)
    return res.json()


def append_to_file(source_id, results, filename):
    with open(filename, 'a') as f:
        for result in results:
            to_write = [
                source_id,
                result['target'],
                result['language'],
                result['pointer']['shortName'],
                result['pointer']['relationGroup'],
                str(result['pointer']['isAutomatic']),
            ]
            f.write('\t'.join(to_write) + '\n')

file_dir = '/Users/annaysun/codenames/babelnet_v4/'
words = ['bear', 'bison', 'jupiter', 'moon', 'phoenix', 'beijing', 'cap', 'boot', 'india', 'germany']

for word in words:
    synsets_capitalized = get_synsets_from_lemma(word.capitalize(), 1)
    synsets_lowered = get_synsets_from_lemma(word.lower(), 3)
    for synset_0 in synsets_capitalized + synsets_lowered:
        results_0 = get_outgoing_edges_json(synset_0)
        append_to_file(synset_0, results_0, file_dir+word)
        hypernyms_0 = get_nonautomatic_hypernyms(results_0)
        for synset_1 in hypernyms_0:
            results_1 = get_outgoing_edges_json(synset_1['target'])
            append_to_file(synset_1['target'], results_1, file_dir+word)
            hypernyms_1 = get_nonautomatic_hypernyms(results_1)
            for synset_2 in hypernyms_1:
                results_2 = get_outgoing_edges_json(synset_2['target'])
                append_to_file(synset_2['target'], results_2, file_dir+word)
                hypernyms_2 = get_nonautomatic_hypernyms(results_2)
