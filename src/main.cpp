
#define SPI_SSD1332                                 // Uncomment if using a SPI SSD1306 OLED, also injects an interp routine for 3x the "frame rate"

#include <driver/i2s.h>
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include "AudioTools.h"

#include "./libraries/kernel/buffer.h"
#include "./libraries/kiss-fft/kiss_fftr.h"
#include "./libraries/kernel/cq_kernel.h"
#include "./libraries/ssd-1322-gfx/SSD1322_for_Adafruit_GFX.h"

//Button configurations
#define BUTTON_BAR_MODE 17                          // Bar Mode Button Selection Pin
#define BUTTON_PEAK_MODE 16                         // Peak Mode Button Selection Pin
#define DEBOUNCE_DELAY 50                           // Mode Button Debounce (ms)

bool barButtonState = HIGH;                         // Current stable state of the Bar Mode button
bool peakButtonState = HIGH;                        // Current stable state of the Peak Mode button
bool lastBarButtonState = HIGH;                     // Previous state of the Bar Mode button
bool lastPeakButtonState = HIGH;                    // Previous state of the Peak Mode button
unsigned long lastDebounceTime = 0;                 // Last debounce time


const adc1_channel_t adc_channel = ADC1_CHANNEL_0;  // Connect DC-biased line signal to this, see IDF docs for pin nums
const enum window_type window_type = GAUSSIAN;      // shape of CQT kernels

// End-user constants, adjust depending on your electrical configuration
const int dB_min = 13;                              // dB, minimum value to display
const int dB_max = 50;                              // dB, maximum value to display
const float fft_mag_cutoff = 70.0;                  // factor used for cutting off noise in raw spectrum, raise if noise is in the output
const int screen_width = 256;                       // px, width of screen
const int screen_height = 64;                       // px, height of screen

// Critical constants, not intended for end-user modification
const int N_samples = 5000;                         // FFT length, prime factorication should contain as many 2s as possible, affects performance
const int sampling_frequency = 44100;               // Hz, I2S sampling frequency
const int max_freq = 6000;                          // Hz, last CQT center freq to display, ensure CQT kernels aren't degenerated when changing
const int min_freq = 38;                            // Hz, first CQT center freq to display, ensure CQT kernels aren't degenerated when changing

const float min_val = 0.08;                         // see Brown CQT paper for explanation
const int calc_rate = 120;                          // Hz, calcs pinned to this rate, artifacts on tone tests and fails to meet calc_rate if too high

const int bar_count_default = 64;
int bar_count = bar_count_default;                  // number of columns to display 16X15, 32x7, 42X5, 64x3
int bar_width = 3;                                  // px, width of each column
const int bar_refresh_rate = 500;
const int peak_hold_delay = 90;                     // Peak hold time lesser holds longer 
const int peak_gap = 2;                             // Gap between in Peak and Bar (px)

int BAR_MODE = 0;                                   // BAR_MODE 0 = Bar + Peak (default), BAR_MODE 1 = Peak, BAR_MODE 2 = Bar
int PEAK_MODE = 0;                                  // PEAK_MODE

// global variables, accessed during execution
struct cq_kernel_cfg cq_cfg = {                     // accessed before all other tasks are started, so its global
    .samples = N_samples,
    .bands = bar_count_default,
    .fmin = min_freq,
    .fmax = max_freq,
    .fs = sampling_frequency,
    .window_type = window_type,
    .min_val = min_val
};

cq_kernels_t kernels;                               // will point to kernels allocated in dynamic memory
int frames; volatile int refresh;                   // benchmarking variables
fftBuffer<float, N_samples> analogBuffer;           // for samples
volatile bool colBuffer_swap_ready = false; doubleBuffer<float, bar_count_default> colBuffer; // for CQT out before post-processing


Adafruit_SSD1322 display(screen_width, screen_height, &SPI, 5, 19, 0);


char bar_position[bar_count_default];
char peak_position[bar_count_default];
char peak_timer[bar_count_default];
unsigned long currentPeakMicros[bar_count_default];

void screen_Task_routine(void *pvParameters){
    colBuffer.alloc();
    display.begin(0x3D);
    display.clearDisplay();
    
    unsigned long currentMicros = micros();
    float   *y = (float*)calloc(bar_count, sizeof(float)),
            *y_1 = (float*)calloc(bar_count, sizeof(float)),
            *y_2 = (float*)calloc(bar_count, sizeof(float)),
            *x_1 = (float*)calloc(bar_count, sizeof(float)),
            *x_2 = (float*)calloc(bar_count, sizeof(float));
    #ifdef SPI_SSD1332
    int cycle_state = 0;
    #endif

    delay(1000); // give time for the other tasks to allocate memory

    while(true){
        #ifdef SPI_SSD1332
        const int interpolation_factor = 2; // 2x the "frame rate"
        const int update_rate = calc_rate*interpolation_factor;
        if(cycle_state == interpolation_factor-1){
            while(!colBuffer_swap_ready); // spin-wait until the buffer is ready
            colBuffer.swap();
            colBuffer_swap_ready = false;
            cycle_state = 0;
        }
        else{
            cycle_state++;
            // while(micros()-currentMicros < 1000000/update_rate);
        }
        // currentMicros = micros();
        #else
        // const int update_rate = refresh_rate;
        // while(micros()-currentMicros < 1000000/refresh_rate);
        // currentMicros = micros();
        if(colBuffer_swap_ready){
            colBuffer.swap();
            colBuffer_swap_ready = false;
        }
        #endif
        
        if(micros()-currentMicros > 100000/bar_refresh_rate){
            currentMicros = micros();
            for(int i = 0; i < bar_count; i++){
                float x = colBuffer.readBuffer[i]-dB_min;
                if(x < 0) x = 0;
                #ifdef SPI_SSD1332
                // upsampling by inserting zeros then filtering, rather than holding x for extended time 
                //  (called zero-order hold) then filtering
                if(cycle_state == 0) x *= interpolation_factor;
                else x = 0;
                // 2nd-order Butterworth IIR with cutoff at 10Hz (240Hz "sampling") as an interpolator
                y[i] = 0.014401440346511*x+0.028802880693022*x_1[i]+0.014401440346511*x_2[i] \
                    +1.632993161855452*y_1[i]-0.690598923241497*y_2[i];
                #else
                // 2nd-order Butterworth IIR with cutoff at 10Hz (89Hz "sampling") as a filter
                y[i] = 0.081926471866054*x+0.163852943732109*x_1[i]+0.081926471866054*x_2[i] \
                    +1.043326781704508*y_1[i]-0.371032669168726*y_2[i];
                #endif

                x_2[i] = x_1[i];
                x_1[i] = x;
                y_2[i] = y_1[i];
                y_1[i] = y[i];
            }
        }

        display.clearDisplay();
        const int col_px = screen_width/bar_count;

        // Bars
        if(PEAK_MODE==0 || PEAK_MODE==1)
        {
            for(int i = 0; i < bar_count; i++){
                int length = y[i]*((float)screen_height/(dB_max-dB_min))-peak_gap;

            if(length<1){
                length = 1;
            }
            
            display.fillRect(i*col_px-bar_width, screen_height-length, bar_width,length, SSD1322_WHITE);
            }
        }
        

         // Peaks
        if(PEAK_MODE==0 || PEAK_MODE==2)
        {
            for(int i = 0; i < bar_count; i++){
            int length = y[i]*((float)screen_height/(dB_max-dB_min));
            if(peak_position[i] < length){
                peak_position[i] = length;
                peak_timer[i] = 1;
                currentPeakMicros[i] = micros();
            }else if(peak_position[i] > length){
                if(peak_timer[i]==0){
                    peak_position[i] -= 1;
                }
            }
            if(micros()-currentPeakMicros[i] > 10000000/peak_hold_delay){
                peak_timer[i] = 0;
            }
            if(peak_position[i]<1){
                peak_position[i] = 1;
            }
            display.drawFastHLine(i*col_px-bar_width, screen_height-peak_position[i], bar_width, SSD1322_WHITE);
            }
        }

        display.display();
        refresh++;

    }
}





void comp_Task_routine(void *pvParameters){
    // Allocate some large arrays
    float *in = (float*)malloc(N_samples*sizeof(float));
    kiss_fft_cpx *out = (kiss_fft_cpx*)malloc(N_samples*sizeof(kiss_fft_cpx));
    kiss_fftr_cfg cfg = kiss_fftr_alloc(N_samples, 0, NULL, NULL);
    kiss_fft_cpx *bands_cpx = (kiss_fft_cpx*)malloc(bar_count*sizeof(kiss_fft_cpx));

    // Initialize I2S sampling
    const int samples_to_read = sampling_frequency/calc_rate+1; // better to want a sample more than to want too fast
    const int i2s_buffer_size = (1 << (int(log2(samples_to_read))+1));
    const i2s_config_t i2s_cfg = {
        .mode = (i2s_mode_t)( I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_ADC_BUILT_IN ),
        .sample_rate = sampling_frequency,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 2,
        .dma_buf_len = i2s_buffer_size,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
    };
    analogBuffer.alloc();
    int16_t *samples_raw = (int16_t*)malloc(sizeof(int16_t)*i2s_buffer_size);
    float *samples = (float*)malloc(sizeof(float)*i2s_buffer_size);
    i2s_driver_install(I2S_NUM_0, &i2s_cfg, 0, NULL);
    i2s_set_adc_mode(ADC_UNIT_1, ADC1_CHANNEL_0);
    i2s_adc_enable(I2S_NUM_0);

    unsigned long last_clipped = 0;

    delay(1000); // give time for the other tasks to allocate memory

    // Initalize benchmark
    unsigned long currentMillis;
    unsigned long timestamps[5] = {0}; // 0 means not recorded yet
    bool benchmark_started = false;
    bool benchmark_posted = false;

    while(true){
        // begin benchmarking over 5 seconds
        if(!benchmark_posted && !benchmark_started){
            refresh = 0;
            frames = 0;
            currentMillis = millis();
            benchmark_started = true;
        }
        
        if(!timestamps[0]) timestamps[0] = micros(); // run-once timestamping

        size_t bytes_read = 0;
        i2s_read(I2S_NUM_0, samples_raw, sizeof(int16_t)*samples_to_read, &bytes_read, portMAX_DELAY); // blocking call

        bool clipped = false;
        int samples_read = bytes_read/sizeof(int16_t);
        for(int i = 0; i < samples_read; i++){
            if(samples_raw[i] > 3873 || samples_raw[i] < 223) clipped = true;
            samples[i] = (float)(samples_raw[i]-2048)*(1.0f/2048.0f);
        }

        for(int i = 0; i < samples_read; i += 2){ // even and odd samples are switched for some reason
            float temp = samples[i];
            samples[i] = samples[i+1];
            samples[i+1] = temp;
        }

        analogBuffer.write(samples, samples_read); // write only 308 samples to the buffer...
        
        float sum = 0, avg;
        analogBuffer.read(in); // ...and read the past N_samples out!
        for(int i = 0; i < N_samples; i++) sum += in[i];
        avg = sum*(1.0f/N_samples);
        for(int i = 0; i < N_samples; i++){
            in[i] -= avg;
            out[i] = (kiss_fft_cpx){0, 0}; // necessary before calling kiss_fftr
        }

        if(!timestamps[1]) timestamps[1] = micros();
      
        kiss_fftr(cfg, in, out);

        if(!timestamps[2]) timestamps[2] = micros();
        
        // Cutting off noise with a threshold inversely proportional to N_samples
        const float minimum_mag = 2048*fft_mag_cutoff/N_samples;
        const float minimum_mag_squared = minimum_mag*minimum_mag;
        for(int i = 0; i < N_samples; i++)
            if(out[i].r*out[i].r+out[i].i*out[i].i < minimum_mag_squared)
                out[i] = (kiss_fft_cpx){0, 0};

        // Convert FFT output to Constant Q output using cq_kernel
        for(int i = 0; i < bar_count; i++) bands_cpx[i] = (kiss_fft_cpx){0, 0};
        apply_kernels(out, bands_cpx, kernels, cq_cfg);

        if(!timestamps[3]) timestamps[3] = micros();

        for(int i = 0; i < bar_count; i++){
            // Finds decibel value of complex magnitude (relative to 1<<14, apparent maximum)
            float mag_squared = bands_cpx[i].r*bands_cpx[i].r+bands_cpx[i].i*bands_cpx[i].i;
            float x = 10.0f*log10(mag_squared); // dB, (squared in ==> 10*log10, not 20*log10), reference level is arbitrary
            colBuffer.writeBuffer[i] = x;
        }
        colBuffer_swap_ready = true;   // Raises flag to indicate buffer is ready to push

        if(!timestamps[4]) timestamps[4] = micros();

        // Outputs benchmark data
        frames++;
        if(millis()-currentMillis > 5000 && benchmark_started && !benchmark_posted){
            benchmark_started = false;
            benchmark_posted = true;
        }
    }
}


void button_Task_routine(void *pvParameters){
    while (true)
    {
        int barReading = digitalRead(BUTTON_BAR_MODE);
        int peakReading = digitalRead(BUTTON_PEAK_MODE);

        // Check if Bar Mode button state changed
        if (barReading != lastBarButtonState) {
            lastDebounceTime = millis();
        }

        // Check if Peak Mode button state changed
        if (peakReading != lastPeakButtonState) {
            lastDebounceTime = millis();
        }

        // Handle debouncing
        if ((millis() - lastDebounceTime) > DEBOUNCE_DELAY) {
            // Handle Bar Mode button press
            if (barReading != barButtonState) {
                barButtonState = barReading;
                if (barButtonState == LOW) {
                    Serial.println("Bar Mode Button Pressed!");

                    BAR_MODE++;
                    if(BAR_MODE == 0){
                        bar_count = 64; 
                        bar_width = 3;
                    }
                    else if(BAR_MODE == 1){
                        bar_count = 42; 
                        bar_width = 5;
                    }
                    else if(BAR_MODE == 2){
                        bar_count = 32; 
                        bar_width = 7;
                    }
                   
                    if(BAR_MODE > 2){
                        BAR_MODE = 0;
                        bar_count = 64; 
                        bar_width = 3;
                    }

                    char peak_position[bar_count];
                    char peak_timer[bar_count];
                    unsigned long currentPeakMicros[bar_count];

                    struct cq_kernel_cfg cq_cfg = {                     // accessed before all other tasks are started, so its global
                        .samples = N_samples,
                        .bands = bar_count,
                        .fmin = min_freq,
                        .fmax = max_freq,
                        .fs = sampling_frequency,
                        .window_type = window_type,
                        .min_val = min_val
                    };
                }
            }

            // Handle Peak Mode button press
            if (peakReading != peakButtonState) {
                peakButtonState = peakReading;
                if (peakButtonState == LOW) {
                    Serial.println("Peak Mode Button Pressed!");

                    PEAK_MODE++;
                    if(PEAK_MODE > 2){
                        PEAK_MODE = 0;
                    }
                }
            }
        }

        // Save the current readings as last states
        lastBarButtonState = barReading;
        lastPeakButtonState = peakReading;
    }
    
}


void setup() {
    Serial.begin(115200);

    // Generate kernels (memory-intensive!) before starting any other tasks
    kernels = generate_kernels(cq_cfg);
    kernels = reallocate_kernels(kernels, cq_cfg);

    pinMode(BUTTON_BAR_MODE, INPUT_PULLUP);
    pinMode(BUTTON_PEAK_MODE, INPUT_PULLUP);
    disableCore0WDT(); // disable the watchdog in order to let screen_Task_routine spin-wait
  
    xTaskCreatePinnedToCore(screen_Task_routine, "screen", 2500, NULL, configMAX_PRIORITIES-1, new TaskHandle_t, 0);
    xTaskCreatePinnedToCore(comp_Task_routine, "comp", 2500, NULL, configMAX_PRIORITIES-1, new TaskHandle_t, 1);
    xTaskCreatePinnedToCore(button_Task_routine, "mode", 2500, NULL, configMAX_PRIORITIES-1, new TaskHandle_t, 1);

}

void loop() {
    vTaskSuspend(NULL); // suspend the arduino loop
    delay(500);
}