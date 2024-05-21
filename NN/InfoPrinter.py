def heading(heading: str, length: int = 20, banner: str = "-", caps: bool = True):
    halfLen = length // 2
    if caps:
        heading = heading.upper()
    print(banner * halfLen + heading + banner * halfLen)


def parameters(
    params: dict, itemSplit: str = "_", keySplit: str = "{}[{}]", name: str = None
):
    print(f"{name}: ", end="")
    if params is None:
        print("None")
        return
    if "{}" not in itemSplit:
        print(itemSplit.join(map(lambda item: keySplit.format(*item), params.items())))
    else:
        print(
            "".join(
                map(
                    lambda x: itemSplit.format(x),
                    map(lambda item: keySplit.format(*item), params.items()),
                )
            )
        )


if __name__ == "__main__":
    heading("hello", 40)
    parameters(None, name="test")
