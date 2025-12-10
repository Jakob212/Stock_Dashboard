# Hashmap
def two_sum(nums, target):
    map = {}

    for index, value in enumerate(nums):
        if target - value in map:
            return [map[target - value], index]
        else:
            map[value] = index
    return False


def contains_duplicate(nums):
    map = {}
    for index, value in enumerate(nums):
        if value in map:
            return True
        else:
            map[value] = index
    return False

#Two Pointer

def is_palindrome(s):
    pointer1 = 0
    pointer2 = len(s) - 1

    while pointer1 < pointer2:
        if s[pointer1] == s[pointer2]:
            pointer1 += 1
            pointer2 -= 1
        else: 
            return False
    return True

def two_sum_sorted(nums, target):
    left = 0
    right = len(nums) - 1
    while left < right:
        if nums[left] + nums[right] == target:
            return (nums[left], nums[right])
        elif nums[left] + nums[right] > target:
            right -= 1
        else:
            left += 1
    return False