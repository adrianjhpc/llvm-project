#ifndef FNACC_DEVICE_H
#define FNACC_DEVICE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int32_t fnacc_get_num_devices(void);
void fnacc_set_device_num(int32_t device);
int32_t fnacc_get_device_num(void);
#ifdef __cplusplus
}
#endif
#endif
