package com.sem.randGen.controller;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class RandomNumbersController {

    @GetMapping("/")
    public String index() {
        return "index";
    }

    @PostMapping("/generate")
    public String generate(@RequestParam int min, @RequestParam int max, Model model) {
        List<Integer> numbers = generateRandomNumbers(min, max);
        int minimum = Collections.min(numbers);
        int maximum = Collections.max(numbers);
        double медиана = calculateMedian(numbers);
        model.addAttribute("numbers", numbers);
        model.addAttribute("minimum", minimum);
        model.addAttribute("maximum", maximum);
        model.addAttribute("median", медиана);
        return "index";
    }

    private List<Integer> generateRandomNumbers(int min, int max) {
        List<Integer> numbers = new ArrayList<>();
        Random random = new Random();
        for (int i = 0; i < 30; i++) {
            numbers.add(random.nextInt(max - min + 1) + min);
        }
        return numbers;
    }

    private double calculateMedian(List<Integer> numbers) {
        Collections.sort(numbers);
        int size = numbers.size();
        if (size % 2 == 0) {
            return (numbers.get(size / 2 - 1) + numbers.get(size / 2)) / 2.0;
        } else {
            return numbers.get(size / 2);
        }
    }
}
