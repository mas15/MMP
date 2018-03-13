import csv

with open('temp.csv', 'w', encoding='utf-8', newline='') as fw:
    writer = csv.writer(fw)
    with open('all_tweets.csv', 'r', encoding='utf8') as fr:
        reader = csv.reader(fr, delimiter=",")
        to_next = ""
        for line in reader:
            id, text, date = line
            if to_next:
                while text[-1] == ".":
                    text = text[:-1]
                text += to_next
                to_next = ""
            if text.startswith("..."):
                to_next = " " + text[3:]
            else:
                writer.writerow((id, text, date))
