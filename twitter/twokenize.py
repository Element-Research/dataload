# -*- coding: utf-8 -*-
'''
Twokenize -- a tokenizer designed for Twitter text in English and some other European languages.
This tokenizer code has gone through a long history:
(1) Brendan O'Connor wrote original version in Python, http://github.com/brendano/tweetmotif
       TweetMotif: Exploratory Search and Topic Summarization for Twitter.
       Brendan O'Connor, Michel Krieger, and David Ahn.
       ICWSM-2010 (demo track), http://brenocon.com/oconnor_krieger_ahn.icwsm2010.tweetmotif.pdf
(2a) Kevin Gimpel and Daniel Mills modified it for POS tagging for the CMU ARK Twitter POS Tagger
(2b) Jason Baldridge and David Snyder ported it to Scala
(3) Brendan bugfixed the Scala port and merged with POS-specific changes
    for the CMU ARK Twitter POS Tagger
(4) Tobi Owoputi ported it back to Java and added many improvements (2012-06)
Current home is http://github.com/brendano/ark-tweet-nlp and http://www.ark.cs.cmu.edu/TweetNLP
There have been at least 2 other Java ports, but they are not in the lineage for the code here.
Ported to Python by Myle Ott <myleott@gmail.com>.
Modified by Y. Wang to tokenize the dataset @ http://help.sentiment140.com/for-students/
Original dataset can be downloaded @ http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
Example comamnd:
    python twokenize.py -i testdata.manual.2009.06.14.csv -o testtokens.manual.2009.06.14.csv
    python twokenize.py -i training.1600000.processed.noemoticon.csv -o trainingtokens.1600000.processed.noemoticon.csv
'''

import re
import csv

import optparse
parser = optparse.OptionParser()

parser.add_option("-i", "--input", action="store", dest="input", type="string", 
   help="Input CSV file", default='')
parser.add_option("-o", "--output", action="store", dest="output", type="string", 
   help="Output CSV file", default='')
parser.add_option("--debug", action="store_true", dest="debug",
   default=False, help="Print debug messages")

(opt, args) = parser.parse_args()
if opt.debug:
   print(opt)

__all__ = ['tokenize']


def regex_or(*items):
    return '(?:' + '|'.join(items) + ')'

contractions = re.compile("(?i)(\w+)(n['’′]t|['’′]ve|['’′]ll|['’′]d|['’′]re|['’′]s|['’′]m)$", re.UNICODE)
whitespace = re.compile("[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)

punct_chars = r"['\"“”‘’.?!…,:;]"
# punct_seq = punct_chars + "+"    # 'anthem'. => ' anthem '.
punct_seq = r"['\"“”‘’]+|[.?!,…]+|[:;]+"   # 'anthem'. => ' anthem ' .
entity = r"[&<>\"]"
# URLs

# BTO 2012-06: everyone thinks the daringfireball regex should be better, but they're wrong.
# If you actually empirically test it the results are bad.
# Please see https://github.com/brendano/ark-tweet-nlp/pull/9

url_start_1 = r"(?:https?://|\bwww\.)"
commonTLDs = r"(?:com|org|edu|gov|net|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|pro|tel|travel|xxx)"
cc_tlds = \
    r"(?:ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|" \
    r"bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|" \
    r"er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|" \
    r"hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|" \
    r"lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|" \
    r"nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|" \
    r"sl|sm|sn|so|sr|ss|st|su|sv|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|" \
    r"va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|za|zm|zw)"   # TODO: remove obscure country domains?
url_start_2 = \
    r"\b(?:[A-Za-z\d-])+(?:\.[A-Za-z0-9]+){0,3}\." + regex_or(commonTLDs, cc_tlds) + r"(?:\." + cc_tlds + r")?(?=\W|$)"
url_body = r"(?:[^\.\s<>][^\s<>]*?)?"
url_extra_crap_before_end = regex_or(punct_chars, entity) + "+?"
url_end = r"(?:\.\.+|[<>]|\s|$)"
url = regex_or(url_start_1, url_start_2) + url_body + "(?=(?:" + url_extra_crap_before_end + ")?" + url_end + ")"

# Numeric
time_like = r"\d+(?::\d+){1,2}"
# num_num = r"\d+\.\d+"
number_with_commas = r"(?:(?<!\d)\d{1,3},)+?\d{3}" + r"(?=(?:[^,\d]|$))"
num_comb = \
    "[\u0024\u058f\u060b\u09f2\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6" \
    "\u00a2-\u00a5\u20a0-\u20b9]?\\d+(?:\\.\\d+)+%?"

# Abbreviations
boundary_not_dot = regex_or("$", r"\s", r"[“\"?!,:;]", entity)
aa1 = r"(?:[A-Za-z]\.){2,}(?=" + boundary_not_dot + ")"
aa2 = r"[^A-Za-z](?:[A-Za-z]\.){1,}[A-Za-z](?=" + boundary_not_dot + ")"
standard_abbreviations = r"\b(?:[Mm]r|[Mm]rs|[Mm]s|[Dd]r|[Ss]r|[Jj]r|[Rr]ep|[Ss]en|[Ss]t)\."
arbitrary_abbrev = regex_or(aa1, aa2, standard_abbreviations)
separators = "(?:--+|―|—|~|–|=)"
#decorations = "(?:[♫♪]+|[★☆]+|[♥❤♡]+|[\u2639-\u263b]+|[\ue001-\uebbb]+)" #this original version removes extra letters
decorations = "(?:[♫♪]+|[★☆]+|[♥❤♡]+)"
things_that_split_words = r"[^\s\.,?\"]"
embedded_apostrophe = things_that_split_words + r"+['’′]" + things_that_split_words + "*"

#  Emoticons
# myleott: in Python the (?iu) flags affect the whole expression
# normal_eyes = "(?iu)[:=]"  # 8 and x are eyes but cause problems
normal_eyes = "[:=]"  # 8 and x are eyes but cause problems
wink = "[;]"
nose_area = "(?:|-|[^a-zA-Z0-9 ])"  # doesn't get :'-(
happy_mouths = r"[D\)\]\}]+"
sad_mouths = r"[\(\[\{]+"
tongue = "[pPd3]+"
other_mouths = r"(?:[oO]+|[/\\]+|[vV]+|[Ss]+|[|]+)"  # remove forward slash if http://'s aren't cleaned

# mouth repetition examples:
# @aliciakeys Put it in a love song :-))
# @hellocalyclops =))=))=)) Oh well

# myleott: try to be as case insensitive as possible, but still not perfect, e.g., o.O fails
# bf_left = "(♥|0|o|°|v|\\$|t|x|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)"
bf_left = "(♥|0|[oO]|°|[vV]|\\$|[tT]|[xX]|;|\u0ca0|@|ʘ|•|・|◕|\\^|¬|\\*)"
bf_center = r"(?:[\.]|[_-]+)"
bf_right = r"\2"
s3 = r"(?:--['\"])"
s4 = r"(?:<|&lt;|>|&gt;)[\._-]+(?:<|&lt;|>|&gt;)"
s5 = "(?:[.][_]+[.])"
# myleott: in Python the (?i) flag affects the whole expression
# basic_face = "(?:(?i)" + bf_left + bf_center + bf_right + ")|" + s3 + "|" + s4 + "|" + s5
basic_face = "(?:" + bf_left + bf_center + bf_right + ")|" + s3 + "|" + s4 + "|" + s5

ee_left = r"[＼\\ƪԄ\(（<>;ヽ\-=~\*]+"
ee_right = "[\\-=\\);'\u0022<>ʃ）/／ノﾉ丿╯σっµ~\\*]+"
ee_symbol = r"[^A-Za-z0-9\s\(\)\*:=-]"
east_emote = ee_left + "(?:" + basic_face + "|" + ee_symbol + ")+" + ee_right

oo_emote = r"(?:[oO]" + bf_center + r"[oO])"


emoticon = regex_or(
    # Standard version  :) :( :] :D :P
    "(?:>|&gt;)?" + regex_or(normal_eyes, wink) + regex_or(nose_area, "[Oo]") +
    regex_or(tongue + r"(?=\W|$|RT|rt|Rt)", other_mouths + r"(?=\W|$|RT|rt|Rt)", sad_mouths, happy_mouths),

    # reversed version (: D:  use positive lookbehind to remove "(word):"
    # because eyes on the right side is more ambiguous with the standard usage of : ;
    regex_or("(?<=(?: ))", "(?<=(?:^))") + regex_or(sad_mouths, happy_mouths, other_mouths) + nose_area +
    regex_or(normal_eyes, wink) + "(?:<|&lt;)?",

    # inspired by http://en.wikipedia.org/wiki/User:Scapler/emoticons#East_Asian_style
    east_emote.replace("2", "1", 1), basic_face,
    # iOS 'emoji' characters (some smileys, some symbols) [\ue001-\uebbb]
    # TODO should try a big precompiled lexicon from Wikipedia, Dan Ramage told me (BTO) he does this

    # myleott: o.O and O.o are two of the biggest sources of differences
    #          between this and the Java version. One little hack won't hurt...
    oo_emote
)

hearts = "(?:<+/?3+)+"  # the other hearts are in decorations

#arrows = regex_or(r"(?:<*[-―—=]*>+|<+[-―—=]*>*)", "[\u2190-\u21ff]+") #this original version removes some extra letters
arrows = r"(?:<*[-―—=]*>+|<+[-―—=]*>*)"

# BTO 2011-06: restored hashtag, at_mention protection (dropped in original scala port) because it fixes
# "hello (#hashtag)" ==> "hello (#hashtag )"  WRONG
# "hello (#hashtag)" ==> "hello ( #hashtag )"  RIGHT
# "hello (@person)" ==> "hello (@person )"  WRONG
# "hello (@person)" ==> "hello ( @person )"  RIGHT
# ... Some sort of weird interaction with edgepunct I guess, because edgepunct
# has poor content-symbol detection.

# This also gets #1 #40 which probably aren't hashtags .. but good as tokens.
# If you want good hashtag identification, use a different regex.
hashtag = "#[a-zA-Z0-9_]+"  # optional: lookbehind for \b
# optional: lookbehind for \b, max length 15
at_mention = "[@＠][a-zA-Z0-9_]+"

# I was worried this would conflict with at-mentions
# but seems ok in sample of 5800: 7 changes all email fixes
# http://www.regular-expressions.info/email.html
bound = r"(?:\W|^|$)"
email = regex_or("(?<=(?:\W))", "(?<=(?:^))") + r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}(?=" + bound + ")"

# We will be tokenizing using these regexps as delimiters
# Additionally, these things are "protected", meaning they shouldn't be further split themselves.
protected = re.compile(
    regex_or(
        hearts,
        url,
        email,
        time_like,
        # num_num,
        number_with_commas,
        num_comb,
        emoticon,
        arrows,
        entity,
        punct_seq,
        arbitrary_abbrev,
        separators,
        decorations,
        embedded_apostrophe,
        hashtag,
        at_mention
    ), re.UNICODE)

# Edge punctuation
# Want: 'foo' => ' foo '
# While also:   don't => don't
# the first is considered "edge punctuation".
# the second is word-internal punctuation -- don't want to mess with it.
# BTO (2011-06): the edgepunct system seems to be the #1 source of problems these days.
# I remember it causing lots of trouble in the past as well.  Would be good to revisit or eliminate.

# Note the 'smart quotes' (http://en.wikipedia.org/wiki/Smart_quotes)
# edge_punct_chars = r"'\"“”‘’«»{}\(\)\[\]\*&"  # add \\p{So}? (symbols)
edge_punct_chars = "'\"“”‘’«»{}\\(\\)\\[\\]\\*&"  # add \\p{So}? (symbols)
edge_punct = "[" + edge_punct_chars + "]"
not_edge_punct = "[a-zA-Z0-9]"  # content characters
off_edge = r"(^|$|:|;|\s|\.|,)"  # colon here gets "(hello):" ==> "( hello ):"
edge_punct_left = re.compile(off_edge + "(" + edge_punct + "+)(" + not_edge_punct + ")", re.UNICODE)
edge_punct_Right = re.compile("(" + not_edge_punct + ")(" + edge_punct + "+)" + off_edge, re.UNICODE)


def split_edge_punct(input_):
    input_ = edge_punct_left.sub(r"\1\2 \3", input_)
    input_ = edge_punct_Right.sub(r"\1 \2\3", input_)
    return input_


# The main work of tokenizing a tweet.
def simple_tokenize(text):
    # Do the no-brainers first
    split_punct_text = split_edge_punct(text)
    if opt.debug:
        print('split_punct_text:', split_punct_text)

    text_length = len(split_punct_text)

    # BTO: the logic here got quite convoluted via the Scala porting detour
    # It would be good to switch back to a nice simple procedural style like in the Python version
    # ... Scala is such a pain.  Never again.

    # Find the matches for subsequences that should be protected,
    # e.g. URLs, 1.0, U.N.K.L.E., 12:53
    bads = []
    bad_spans = []
    for match in protected.finditer(split_punct_text):
        # The spans of the "bads" should not be split.
        if match.start() != match.end():  # unnecessary?
            bads.append([split_punct_text[match.start():match.end()]])
            bad_spans.append((match.start(), match.end()))
    if opt.debug:
        print('bads:', bads)

    # Create a list of indices to create the "goods", which can be
    # split. We are taking "bad" spans like
    #     List((2,5), (8,10))
    # to create
    #     List(0, 2, 5, 8, 10, 12)
    # where, e.g., "12" here would be the text_length
    # has an even length and no indices are the same
    indices = [0]
    for (first, second) in bad_spans:
        indices.append(first)
        indices.append(second)
    indices.append(text_length)

    # Group the indices and map them to their respective portion of the string
    split_goods = []
    for i in range(0, len(indices), 2):
        good_str = split_punct_text[indices[i]:indices[i + 1]]
        split_str = good_str.strip().split(" ")
        split_goods.append(split_str)
    if opt.debug:
        print('split_goods:', split_goods)

    # Reinterpolate the 'good' and 'bad' Lists, ensuring that
    # additonal tokens from last good item get included
    zipped_str = []
    for i in range(len(bads)):
        zipped_str = add_all_nonempty(zipped_str, split_goods[i])
        zipped_str = add_all_nonempty(zipped_str, bads[i])
    zipped_str = add_all_nonempty(zipped_str, split_goods[len(bads)])

    # BTO: our POS tagger wants "ur" and "you're" to both be one token.
    # Uncomment to get "you 're"
    split_str = []
    for tok in zipped_str:
        split_str.extend(split_token(tok))
    zipped_str = split_str
    if opt.debug:
        print('zipped_str:', zipped_str)

    return zipped_str


def add_all_nonempty(master, smaller):
    for s in smaller:
        strim = s.strip()
        if len(strim) > 0:
            master.append(strim)
    return master


# "foo   bar " => "foo bar"
def squeeze_whitespace(input_):
    #outstr = whitespace.sub(" ", input_).strip() #this original version removes much more than extra white space
    outstr = ' '.join(input_.split()).strip()
    return outstr


# Final pass tokenization based on special patterns
def split_token(token):
    m = contractions.search(token)
    if m:
        return [m.group(1), m.group(2)]
    return [token]


# Assume 'text' has no HTML escaping.
def tokenize(text):
    text2 = squeeze_whitespace(text)
    if opt.debug:
        print('squeeze_whitespace', text2)
    return simple_tokenize(text2)

if __name__ == '__main__':
    csv_data = []
    with open(opt.input) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            if opt.debug:
                print(row)
            tokens = tokenize(row[-1])
            row[-1] = ' '.join(tokens)
            csv_data.append(row)
    if len(csv_data)>0 and opt.output!='':
        with open(opt.output, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', doublequote=True, quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writerows(csv_data)
            #for row in csv_data:
            #   csvfile.writelines('"%s"\n'%('","'.join(row)))
    else:
        print('empty output')
