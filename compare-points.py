import csv
import sys
import os

total_matched = 0;
total_missed = 0;

def take_query_pid(elem):
    return elem[0]

def parse_csv(path):
    with open(path, 'r') as csv_file:
        results = []
        reader = csv.reader(csv_file)
        next(reader) # skip header
        for row in reader: # each row is a list
            results.append(row)
        return results

# returns True if all pairs in rows_a has a corresponding pair in rows_b
def does_csv_match(rows_a, rows_b):
    global total_matched
    global total_missed
    all_matched = True
    
    for i in range(0, len(rows_a)):
        matched = True
        rowa = rows_a[i]
        rowb = rows_b[i]
        for j in range(0, len(rowa)):
            if (rowa[j] != rowb[j]):
                matched = False
    # for rowa in rows_a:
    #     pid1 = rowa[0]
    #     pid2 = rowa[1]
    #     matched = True
    #     for rowb in rows_b:
            # if (rowb[0] == pid1 and rowb[1] == pid2) or (rowb[0] == pid2 and rowb[1] == pid1):
            #     if rowa[2] == rowb[2] and rowa[3] == rowb[3] and rowb[4] == rowb[4] and rowb[5] == rowb[5]:
            #         matched = True
            #         total_matched += 1
            #     else:
            #         # print(rowa[2])
            #         # print(rowb[2])
            #         print("Matched, but different computed values for " + pid1 + ", " + pid2)
            #         # return
            #         matched = True
            #         total_matched += 1
        if not matched:
            print("Did not find a match for " + pid1 + ", " + pid2)
            all_matched = False
            total_missed += 1
        else:
            total_matched += 1
    return all_matched

def compare_csvs(num):
    path_a = sys.argv[1] + '/particles_' + str(num) + '.csv'
    path_b = sys.argv[2] + '/particles_' + str(num) + '.csv'

    rows_a = parse_csv(path_a)
    rows_b = parse_csv(path_b)

    matched = True

    if not len(rows_a) == len(rows_b):
        print("Different lengths: " + path_a + " and " + path_b)
        matched = False

    if not does_csv_match(rows_a, rows_b):
        print("Problem with comparing " + path_a + " to " + path_b)
        matched = False
        
    if not does_csv_match(rows_b, rows_a):
        print("Problem with comparing " + path_b + " to " + path_a)
        matched = False
    # if matched:
        # print("no problems with " + str(num))

    return matched
    
def compare_all_csvs():
    csvs_count_1 = len(os.listdir(sys.argv[1]))
    csvs_count_2 = len(os.listdir(sys.argv[2]))
    all_matched = True

    for i in range(1, min(csvs_count_1, csvs_count_2)):
        if not compare_csvs(i):
            all_matched = False
        elif i % 100 == 0:
            print("No problems from " + str(i - 100) + " to " + str(i))

    if all_matched:
        print("All CSVs matched!")
    else:
        print("There were some problems")

    print("Matched: {matched} | Missed: {missed}".format(matched=total_matched / 2, missed=total_missed))

compare_all_csvs()

# compare_csvs(13)