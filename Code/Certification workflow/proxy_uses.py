

def confidence_threshold_proxy_use(proxy_congestion_probability, proxy_upper_confidence_threshold, proxy_lower_confidence_threshold):
    proxy_can_be_used = (proxy_congestion_probability > proxy_upper_confidence_threshold) or (proxy_congestion_probability < proxy_lower_confidence_threshold)
    return proxy_can_be_used
