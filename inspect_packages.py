from importlib.metadata import distribution

for package in ("numpy", "scipy", "tensorflow"):
    dist = distribution(package)
    print(package, dist.version, dist.locate_file(""))
