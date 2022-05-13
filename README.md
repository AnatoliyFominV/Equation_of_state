"# Equation_of_state" 
# 2 Phase flash calculations

Расчёт компонентного сотава системы жидкость - газ. По уравнению состояния Пенга-Робинсона.

Результаты расчёта:
Константы фазового равновесия - для каждго компонента.
Расчёт плотности / Обьемного коэффициента / Газосодержания.
Построение графика этих параметров.

*Расчёт стабилен, только если в данных условиях существуют 2 фазы.*


# Схема расчётов

Рассмотрим порядок выполнения расчётов.

Входные данные: 
* Глобальные мольные доли компонентов,
* Молекулярная масса,
* Критическая температура,
* Критическое давление, 
* Критический обьём,
* Ацентрический фактор,
* Коэффициенты попарного взаимодействия.
* Давление 
* Температура
* Точность расчётов

Поскольку мы хотим запускать расчёт множество раз, при разных давлениях. То создадим цикл **for P in P_all** в котором будем высчитывать константы фазового равновесия для каждого заданного давления.
Поскольку константы подбираются итеративно, требуется начальное приближение, которые подбирается из уравнения Вильсона:
![alt text for screen readers](/image/Wilson.png "Wilson eq")
![enter image description here](https://disk.yandex.ru/i/unT15DtApIdTWQ)

Затем константы итеративно обновляются, домножаясь на отношение летучести жидкости к летучести газа.
 
## MultiPhazeCalc

Имея начально предположение о константах равновесия, мы можем расчитать летучести фаз ( а они должны быть равны ), и если они не равны, то уточнить константы равновесия.

Но для этого нам придётся пройти по следующим этапам

### 1.  Материальный баланс

Зная константы равновесия для каждого компонента, мы можем подобрать мольную долю газовой фазы в смеси из уравнения материального баланса:

![alt text for screen readers](/image/Mat_balance.png "Mat_balance eq")

Поскольку нам известна глобальная мольная доля газа, то мы можем расчитать мольные доли компонентов в газовой и жидкой фазе. И к каждой по отдельности применить уравнение состояния.

### 2. Уравнение состояния

Уравнение состояния выраженное через коэффициент сверхсжимаемости Z:
![alt text for screen readers](/image/Eq_state.png "Eq_state")
Где коэффициенты A и B:
![alt text for screen readers](/image/A_B.png "A_B")
Для которых коэффициенты a, b:
![alt text for screen readers](/image/A_B.png "a2_b2")

Подробнее про их расчёт можно найти в документе: Cubic_Equation_of_State

Решить уравнение можно 2 способами 
1) Подобрать решение численно. Реализовано: Z_calculate
2) Решить аналитически по формуле реализованной в eos

Теперь у нас есть 3 корня - коэффициента сверхсжимаемости, для каждой фазы, но правильный только 1. Поэтому необходимо выбрать из них правильный.

### 3. Подбор корней

Первым делом проверяем действительно ли число является корнем уравнения функцией Z_check.

Затем выбираем то коэффициента сверхсжимаемости, с котором у фазы  наименьшее значение энергии Гибса. Функция Z_Choose_MinG

### 4. Проверка точности констант фазового равновесия

Решив уравнение состояния и выбрав верный корень, для каждой фазы. Мы можем посчитать летучести и сравнить их.
Формула логарифма коэффициента летучести:
![alt text for screen readers](/image/fug.png "fug")

Чтобы полученть летучесть требуется его возвести в exp, домножить на мольную долю компонентов в фазе и на давление. 

Просчитав разность между ними вернём её как погрешность

## Расчёт параметров 

После того, как мы определили константу равносия, мы можем определить обьёмы, плотности фаз и газосодержание жидкости.
Которые можно вывести на графике.
