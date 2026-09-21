from climate_funcs import *

def main():
    dataset, filled_dataset = create_dataset()

    print("==== Task 1 ====")
    tropical, freezing, pleasant = count_special_days(filled_dataset)
    print(f"The tropical day count for the first three cities are: {tropical[:3]}")
    print(f"The freezing day count for the first three cities are: {freezing[:3]}")
    print(f"The pleasant day count for the first three cities are: {pleasant[:3]}")

    most_extreme_city = find_most_extreme_city(tropical, freezing)
    print(f"The city with the most extreme weather is {most_extreme_city + 1}.")

    print("==== Task 2 ====")
    heatwave_stat, longest_heatwave = compute_heatwave_stats(filled_dataset)
    print(f"The heatwave count for the first three cities are: {heatwave_stat[:3]}")
    print(f"The city with the longest heatwave is City {longest_heatwave[0] + 1}, with a heatwave lasting {longest_heatwave[1]} day(s).")

    print("==== Task 3 ====")
    warm_periods = count_warm_periods(filled_dataset)
    print(f"The number of 7-day warm periods in the first three cities are: {warm_periods[:3]}")

    largest_increase, largest_decrease = find_sharpest_changes(filled_dataset)
    print(f"The month with the largest increase in temperature for the first three cities are: {largest_increase[:3] + 1}")
    print(f"The month with the largest decrease in temperature for the first three cities are: {largest_decrease[:3] + 1}")

    print("==== Task 4 ====")
    record_high, record_low, rapid_changes = find_record_breaking_temperatures(filled_dataset)
    print(f"The number of record-breaking high temperatures for the first three cities are: {record_high[:3]}")
    print(f"The number of record-breaking low temperatures for the first three cities are: {record_low[:3]}")
    print(f"The number of rapid temperature changes for the first three cities are: {rapid_changes[:3]}")

    threshold_crosses = find_threshold_crossings(filled_dataset)
    print(f"The number of threshold crossings with threshold = 20°C for the first three cities are: {threshold_crosses[:3]}")

    print("==== Task 5 ====")
    histogram = convert_to_histogram(dataset)
    print("The histogram for the first three cities are:")
    print(histogram[:, :3])

    first, second = find_correlated_pair(dataset)
    print(f"The most correlated pair of cities are City {first + 1} and {second + 1}")

    print("==== Task 6 ====")
    heat_spikes, cold_snaps = count_extreme_weather_patterns(filled_dataset)
    print(f"The number of heat spikes in the first three cities are: {heat_spikes[:3]}")
    print(f"The number of cold snaps in the first three cities are: {cold_snaps[:3]}")

if __name__ == "__main__":
    main()
