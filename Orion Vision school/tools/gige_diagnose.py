def diagnose_features(cam) -> None:
    """Print alle beschikbare bandbreedte-gerelateerde features."""

    features_to_check = [
        "StreamBytesPerSecond",
        "DeviceLinkThroughputLimit",
        "DeviceLinkSpeed",
        "DeviceLinkCurrentThroughput",
        "GevSCPSPacketSize",
        "GevSCPD",
        "GVSPAdjustPacketSize",
        "StreamBufferHandlingMode",
        "AcquisitionFrameRate",
        "AcquisitionFrameRateEnable",
    ]

    print("\n[FEATURE SCAN]")
    print(f"  {'Feature':<35} {'Beschikbaar':<12} {'Waarde'}")
    print(f"  {'-'*35} {'-'*12} {'-'*20}")

    for name in features_to_check:
        try:
            feat = cam.get_feature_by_name(name)
            try:
                value = feat.get()
                try:
                    rng = feat.get_range()
                    info = f"{value}  [range: {rng[0]}–{rng[1]}]"
                except Exception:
                    info = str(value)
            except Exception:
                info = "(write-only / command)"
            print(f"  {'✓ ' + name:<35} {'JA':<12} {info}")
        except Exception:
            print(f"  {'✗ ' + name:<35} {'nee':<12}")

# Gebruik in diagnose():
#   with cam as c:
#       diagnose_features(c)