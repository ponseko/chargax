import jax.numpy as jnp
import jax_datetime as jdt


def year_and_doy(dt: jdt.Datetime):
    """Convert a datetime to year and day of year (doy).
    This calculation is simplified and assumes the datetime is within the 21st century"""
    base = jdt.to_datetime("2020-01-01")
    days = (dt - base).days  # days since 2020-01-01

    # 4-year cycle: leap(366) + normal(365) + normal(365) + normal(365)
    cycle, remainder = jnp.divmod(days, 1461)

    is_leap_year = remainder < 366
    remainder_in_cycle = remainder - 366

    year_offset = jnp.where(is_leap_year, 0, remainder_in_cycle // 365 + 1)
    doy = jnp.where(is_leap_year, remainder, remainder_in_cycle % 365) + 1
    year = 2020 + cycle * 4 + year_offset

    return year, doy
