// Press Shift twice to open the Search Everywhere dialog and type `show whitespaces`,
// then press Enter. You can now see whitespace characters in your code.
//public class Main {
//    public static void main(String[] args) {
//                int[] numbers = {12, 34, 56, 78, 90, 45, 23};
//
//                int largest = findLargestElement(numbers);
//
//                System.out.println("Наибольший элемент: " + largest);
//            }
//
//            public static int findLargestElement(int[] arr) {
//                int largest = Integer.MIN_VALUE;
//
//                for (int i = 0; i < arr.length; i++) {
//                    if (arr[i] > largest) {
//                        largest = arr[i];
//                    }
//                }
//
//                return largest;
//            }
//}

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        int[] uniqueNums = removeDuplicates(new int[]{-1, 0, 0, 0, 1, 1, 2});
        for (int j : uniqueNums) {
            System.out.print(j + " ");
        }
    }

    public static int[] removeDuplicates(int[] nums) {
        int j = 1;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] != nums[i - 1]) {
                nums[j] = nums[i];
                j++;
            }
        }
        return Arrays.copyOfRange(nums, 0, j);
    }
}
