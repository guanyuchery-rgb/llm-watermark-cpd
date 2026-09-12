# Unmodified statistic definitions from the pre-refactor 3-detect.py.
if args.method == "transform":
    test_stats = []
    def dist1(x, y): return transform_edit_score(x, y, gamma=args.gamma)

    def test_stat1(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, transform_key_func, dist1, null=False, normalize=True
        )
    test_stats.append(test_stat1)
    def dist2(x, y): return transform_score(x, y)

    def test_stat2(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, transform_key_func, dist2, null=False, normalize=True
        )
    test_stats.append(test_stat2)

elif args.method == "gumbel":
    test_stats = []
    def dist1(x, y): return gumbel_edit_score(x, y, gamma=args.gamma)

    def test_stat1(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, gumbel_key_func, dist1, null=null, normalize=False
        )
    test_stats.append(test_stat1)
    def dist2(x, y): return gumbel_score(x, y)

    def test_stat2(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, gumbel_key_func, dist2, null=null, normalize=False
        )
    test_stats.append(test_stat2)
else:
    raise
