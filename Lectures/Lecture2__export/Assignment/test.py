import matplotlib.pyplot as plt

res = [(0.37307578325271606, 0.8423827290534973), (0.37044215202331543, 0.841964840888977), (0.373121976852417, 0.8419005274772644), (0.3832526206970215, 0.8372713327407837)]
losses = [x[0] for x in res]
accs = [x[1] for x in res]
x = range(1, len(res) + 1)
plt.figure(figsize=(8, 5))
plt.plot(x, losses, marker="o", label="Loss")
plt.plot(x, accs, marker="s", label="Accuracy")
plt.xlabel("Experiment Index")
plt.ylabel("Value")
plt.title("Loss & Accuracy Comparison")
plt.show()
