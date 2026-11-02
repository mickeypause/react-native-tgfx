# Tilt Pen offscreen autorelease-pool comparison

Media for mickeypause/react-native-tgfx PR #1.

Both sides run tgfx 0.0.2-alpha.8 in Tilt Pen. The worker autorelease pool is disabled before and enabled after; the separate startup compatibility patch is retained on both. These are original recordings from the same session, aligned to one scripted 12.6-second drawing sequence after warmup.

The overlay uses recorded host process RSS and UI callback counts. It does not measure GPU presentation FPS or GPU-only memory. The before segment begins with memory already accumulated. This app demonstration is separate from the current-library regression test in the PR.

See media/manifest.json for measured values, conditions and checksums.
