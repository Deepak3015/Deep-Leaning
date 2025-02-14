from collections import Counter

n = int(input().strip())
string = [input().strip()for _ in range(n)]
k = int(input().strip())

count = Counter(string)
