#ifndef TILEOFFLOAD_DEVICE_H
#define TILEOFFLOAD_DEVICE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
int32_t tileoff_get_num_devices(void);
void tileoff_set_device_num(int32_t device);
int32_t tileoff_get_device_num(void);
#ifdef __cplusplus
}
#endif

#endif /* TILEOFFLOAD_DEVICE_H */
