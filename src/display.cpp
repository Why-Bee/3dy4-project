#include <iostream>
#include <ncurses.h>
#include <thread>
#include <mutex>
#include <random>
#include <chrono>

// Global variables
int largest_number = 0;
std::mutex mtx;

// Function to update data
void update_data() {
    while (true) {
        // Generate a random number
        // largest_number++;

        // Update the largest number if the generated number is larger
        std::lock_guard<std::mutex> guard(mtx);
        // if (random_number > largest_number) {
            largest_number++;// = random_number;
        // }

        // Sleep for a short interval
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Function to display data
void display_data() {
    initscr(); // Initialize ncurses
    while (true) {
        clear(); // Clear the screen
        mvprintw(0, 0, "Displaying Data:");
        mvprintw(1, 0, "Largest Number So Far: %d", largest_number);
        refresh(); // Refresh the screen
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    endwin(); // End ncurses
}

int main() {
    // Seed the random number generator
    srand(time(nullptr));

    // Create threads
    std::thread update_thread(update_data);
    std::thread display_thread(display_data);

    // Wait for threads to finish
    update_thread.join();
    display_thread.join();

    return 0;
}
