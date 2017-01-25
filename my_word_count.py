
import sys
from pyspark import SparkContext
from operator import add

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "Usage: my_word_count <file>"
        sys.exit(-1)

    sc = SparkContext(appName='my_word_count')
    text_file = sc.textFile(sys.argv[1], 1).cache()
    words = text_file.flatMap(lambda line: line.split(' '))

    stop_words = ["", "a", "an", "the", "this", "to", "for", "and", "##", "can", "on", "is", "in", "of", "also", "if",
                  "with", "you", "or"]
    punct = "\"'`,:.![]<>-"

    wc = words.map(lambda raw: raw.strip(punct)).filter(lambda w: w not in stop_words).map(lambda w: (w,1)).reduceByKey(add)

    for (w, c) in wc.top(10, lambda x: x[1]):
        print "[%s, %i]" % (w, c)

    sc.stop()




















