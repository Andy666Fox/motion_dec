import numpy as np 
import cv2

camera = cv2.VideoCapture(0)

while True:
    iSee = False     # флаг: был ли найден контур
    controlX = 0.0      # нормализованное отклонение цветного объекта от центра кадра в диапазоне [-1; 1]
    
    success, frame = camera.read()  # читаем кадр с камеры
    
    if success:     # если прочитали успешно
        height, width = frame.shape[0:2]    # получаем разрешение кадра
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)    # переводим кадр из RGB в HSV
        #binary = cv2.inRange(hsv, (18, 60, 100), (32, 255, 255))  # пороговая обработка кадра (выделяем все желтое)
        binary = cv2.inRange(hsv, (90, 50, 50), (120, 255, 255))  # пороговая обработка кадра (выделяем все синее)
        
        roi = cv2.bitwise_and(frame, frame, mask=binary)    # за счет полученной маски можно выделить найденный объект из общего кадра

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)  # получаем контуры выделенных областей 

        if len(contours) != 0:  # если найден хоть один контур
            maxc = max(contours, key=cv2.contourArea)  # находим наибольший контур
            moments = cv2.moments(maxc) # получаем моменты этого контура
            """
            # moments["m00"] - нулевой момент соответствует площади контура в пикселях,
            # поэтому, если в битовой маске присутствуют шумы, можно вместо
            # if moments["m00"] != 0:  # использовать
                
            if moments["m00"] > 20: # тогда контуры с площадью меньше 20 пикселей не будут учитываться 
            """
            if moments["m00"] > 20: #  контуры с площадью меньше 20 пикселей не будут учитываться 
                cx = int(moments["m10"] / moments["m00"])  # находим координаты центра контура (найденного объекта) по x
                cy = int(moments["m01"] / moments["m00"])  # находим координаты центра контура (найденного объекта) по y

                iSee = True  # устанавливаем флаг, что контур найден
                controlX = 2 * (cx - width/2) / width   # находим отклонение найденного объекта от центра кадра и нормализуем его (приводим к диапазону [-1; 1])
        
                cv2.drawContours(frame, maxc, -1, (0, 255, 0), 2)  # рисуем контур
                cv2.line(frame, (cx, 0), (cx, height), (0, 255, 0), 2)  # рисуем линию линию по x
                cv2.line(frame, (0, cy), (width, cy), (0, 255, 0), 2)  # линия по y
                
        cv2.putText(frame, 'iSee: {};'.format(iSee), (width - 370, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA) # текст
        cv2.putText(frame, 'controlX: {:.2f}'.format(controlX), (width - 200, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA) # текст
                

        #cv2.circle(frame, (width//2, height//2), radius=4, color=(0, 0, 255), thickness=-1)    # центр кадра
        cv2.imshow('frame', frame)  # выводим все кадры на экран
        #cv2.imshow('binary', binary)
        #cv2.imshow('roi', roi)
        
    if cv2.waitKey(1) == ord('q'):  # чтоб выйти надо нажать 'q' на клавиатуре
        break

camera.release()
cv2.destroyAllWindows()

    