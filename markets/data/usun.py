import csv
#
# with open('temp.csv', 'w', encoding='utf-8', newline='') as fw:
#     writer = csv.writer(fw)
#     with open('all_tweets.csv', 'r', encoding='utf8') as fr:
#         reader = csv.reader(fr, delimiter=",")
#         prev_id = False
#         prev_text = ""
#         for line in reader:
#             id, text, date = line
#
#             if text.endswith("...") and not text.startswith("RT"):
#                 while text[-1] == ".":
#                     text = text[:-1]
#
#                 print()
#                 print(text)
#                 print(prev_text)
#                 print()
#
#                 text += " " + prev_text
#                 prev_id, prev_text, prev_date = id, text, date
#             else:
#                 if prev_id:
#                     writer.writerow((prev_id, prev_text, prev_date))
#                 prev_id, prev_text, prev_date = id, text, date

at = None
at2 = None
with open("attr_to_Remove", "r") as f:
    f.readline()
    at = [attr for attr in f.readlines()]
with open("attr_to_Remove_2", "r") as f:
    f.readline()
    at2 = [attr for attr in f.readlines()]
for a in at2:
    if a not in at:
        print(a)
#print("WSPOLNE " + str(len([a for a in at if a in at2])))
