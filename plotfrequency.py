from matplotlib import pyplot as plt
import json

frequency = json.loads(open("./frequency.data").read())

plt.bar(range(1, 29), [frequency[str(i)]
                       for i in range(1, 29)], width=0.9, color="blue")
plt.xlabel("number of words in name")
plt.ylabel("fraction of words in name appearing in description")
plt.show()
