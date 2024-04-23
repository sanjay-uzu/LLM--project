import datetime
from typing import List, Tuple


def read_requirements(file_path: str) -> List[str]:
    with open(file_path, "r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements


def split_time_range_into_intervals(
    from_datetime: datetime.datetime, to_datetime: datetime.datetime, n: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:

    # Calculate total duration between from_datetime and to_datetime.
    total_duration = to_datetime - from_datetime

    # Calculate the length of each interval.
    interval_length = total_duration / n

    # Generate the interval.
    intervals = []
    for i in range(n):
        interval_start = from_datetime + (i * interval_length)
        interval_end = from_datetime + ((i + 1) * interval_length)
        if i + 1 != n:
            # Subtract 1 microsecond from the end of each interval to avoid overlapping.
            interval_end = interval_end - datetime.timedelta(minutes=1)

        intervals.append((interval_start, interval_end))

    return intervals
